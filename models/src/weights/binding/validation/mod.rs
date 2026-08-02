use std::collections::BTreeMap;

use super::{
    BindingTransform, BlockFormat, BlockProjectionLayout, BlockQuantization, LayerTensorRole,
    LinearAttentionTensorRole, LogicalTensorRole, TensorBinding, TensorStorage, WeightBindingPlan,
};
use crate::{
    error::{ModelsError, Result},
    weights::TensorCatalog,
};

mod affine;
mod awq;
mod bitsandbytes;
mod float8;
mod gptq;
mod packed_integer;

pub(super) fn validate(plan: &WeightBindingPlan, catalog: &TensorCatalog) -> Result<()> {
    unique_roles(plan)?;
    for binding in &plan.tensors {
        let Some(logical) = binding.logical_shape.as_deref() else {
            continue;
        };
        match &binding.storage {
            TensorStorage::Dense { dtype, bias } => {
                let expected = dense_shape(binding, logical);
                if !alternate_convolution(binding, &expected) {
                    shape(&binding.source, &binding.shape, &expected)?;
                }
                if let Some(bias) = bias
                    && logical.len() > 1
                {
                    companion(catalog, bias, &logical[..logical.len() - 1])?;
                    companion_dtype(catalog, bias, dtype)?;
                }
            },
            TensorStorage::AffineQuantized { format, scales, biases, output_bias, .. } => {
                affine::validate(binding, logical, *format, scales, biases.as_deref(), catalog)?;
                if let Some(output_bias) = output_bias
                    && logical.len() > 1
                {
                    companion(catalog, output_bias, &logical[..logical.len() - 1])?;
                }
            },
            TensorStorage::PackedInt8 { .. } | TensorStorage::PackedInt4 { .. } => {
                packed_integer::validate(binding, logical, &binding.storage, catalog)?;
            },
            TensorStorage::Awq { format, scales, zero_points } => {
                awq::validate(binding, logical, *format, scales, zero_points, catalog)?;
            },
            TensorStorage::Gptq {
                format,
                scales,
                zero_points,
                group_indices,
            } => {
                gptq::validate(
                    binding, logical, *format, scales, zero_points, group_indices, catalog,
                )?;
            },
            TensorStorage::BitsAndBytes4Bit { .. } => {
                bitsandbytes::validate(binding, logical, catalog)?;
            },
            TensorStorage::Float8 { .. } => float8::validate(binding, logical, catalog)?,
            TensorStorage::BlockQuantized {
                format,
                scales,
                global_scale,
                input_scale,
                bias,
                ..
            } => block(
                binding,
                logical,
                *format,
                scales,
                global_scale.as_deref(),
                input_scale.as_deref(),
                bias.as_deref(),
                catalog,
            )?,
            TensorStorage::Auxiliary { .. } => shape(&binding.source, &binding.shape, logical)?,
        }
    }
    Ok(())
}

fn alternate_convolution(binding: &TensorBinding, expected: &[usize]) -> bool {
    matches!(
        binding.role,
        LogicalTensorRole::Layer {
            tensor: LayerTensorRole::LinearAttention {
                tensor: LinearAttentionTensorRole::Convolution
            },
            ..
        }
    ) && matches!(expected, [channels, kernel, 1] if binding.shape == [*channels, 1, *kernel])
}

fn dense_shape(binding: &TensorBinding, logical: &[usize]) -> Vec<usize> {
    if binding.transforms.contains(&BindingTransform::Transpose) {
        let mut physical = logical.to_vec();
        let last = physical.len() - 1;
        physical.swap(last - 1, last);
        physical
    } else {
        logical.to_vec()
    }
}

fn unique_roles(plan: &WeightBindingPlan) -> Result<()> {
    let mut roles: BTreeMap<&LogicalTensorRole, &str> = BTreeMap::new();
    for binding in &plan.tensors {
        if let Some(previous) = roles.insert(&binding.role, &binding.source) {
            return Err(ModelsError::InvalidConfig(format!(
                "logical tensor role {:?} is bound by both {previous} and {}",
                binding.role, binding.source
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn block(
    binding: &TensorBinding,
    logical: &[usize],
    format: BlockQuantization,
    scales: &str,
    global_scale: Option<&str>,
    input_scale: Option<&str>,
    bias: Option<&str>,
    catalog: &TensorCatalog,
) -> Result<()> {
    validate_block_layout(binding, logical)?;
    let (prefix, input) = projection(logical, &binding.source)?;
    let (divisor, tail) = match (format.format, format.storage_dtype) {
        (BlockFormat::MxFp4, super::BlockStorageDType::U8) => (32, Some(16)),
        (BlockFormat::MxFp4, super::BlockStorageDType::U32) => (8, None),
        (BlockFormat::MxFp4, _) => {
            return Err(invalid(&binding.source, "unsupported MXFP4 container dtype"));
        },
        (BlockFormat::MxFp8, _) => (4, None),
        (BlockFormat::NvFp4, _) => (2, None),
    };
    let columns = divided(input, divisor, &binding.source)?;
    let mut physical = prefix.to_vec();
    physical.push(columns);
    if let Some(tail) = tail {
        physical.push(tail);
    }
    shape(&binding.source, &binding.shape, &physical)?;
    companion_dtype(catalog, &binding.source, format.storage_dtype.as_str())?;
    let scale_divisor = format.block_size;
    let mut scale_shape = prefix.to_vec();
    scale_shape.push(divided(input, scale_divisor, &binding.source)?);
    companion(catalog, scales, &scale_shape)?;
    companion_dtype(catalog, scales, format.block_scale.storage_dtype.as_str())?;
    if format.global_scale.is_some() != global_scale.is_some()
        || format.input_scale.is_some() != input_scale.is_some()
    {
        return Err(invalid(&binding.source, "scale hierarchy differs from block contract"));
    }
    if let Some(global) = global_scale {
        companion(catalog, global, &[])?;
        let expected =
            format.global_scale.ok_or_else(|| invalid(global, "unexpected global scale"))?;
        companion_dtype(catalog, global, expected.storage_dtype.as_str())?;
    }
    if let Some(input_scale) = input_scale {
        companion(catalog, input_scale, &[])?;
        let expected = format
            .input_scale
            .ok_or_else(|| invalid(input_scale, "unexpected input scale"))?;
        companion_dtype(catalog, input_scale, expected.storage_dtype.as_str())?;
    }
    if let Some(bias) = bias {
        companion(catalog, bias, prefix)?;
        let dtype = format
            .output_bias_dtype
            .ok_or_else(|| invalid(bias, "block contract does not admit an output bias"))?;
        companion_dtype(catalog, bias, dtype.as_str())?;
    }
    Ok(())
}

fn validate_block_layout(binding: &TensorBinding, logical: &[usize]) -> Result<()> {
    match binding.block_projection_layout() {
        Some(BlockProjectionLayout::Matrix) if logical.len() == 2 => Ok(()),
        Some(BlockProjectionLayout::MatrixBank { matrices }) if matches!(logical, [actual, _, _] if *actual == matrices) => {
            Ok(())
        },
        Some(BlockProjectionLayout::FusedGateUpBank { experts, .. }) if matches!(logical, [actual, output, _] if *actual == experts && output.is_multiple_of(2)) => {
            Ok(())
        },
        _ => Err(invalid(
            &binding.source,
            "block projection transforms, packing, and logical shape disagree",
        )),
    }
}

pub(super) fn projection<'a>(logical: &'a [usize], source: &str) -> Result<(&'a [usize], usize)> {
    logical
        .split_last()
        .map(|(input, prefix)| (prefix, *input))
        .ok_or_else(|| invalid(source, "projection shape is empty"))
}

pub(super) fn divided(value: usize, divisor: usize, source: &str) -> Result<usize> {
    value
        .checked_div(divisor)
        .filter(|result| value.is_multiple_of(divisor) && *result > 0)
        .ok_or_else(|| invalid(source, "block geometry does not divide logical input"))
}

pub(super) fn companion(catalog: &TensorCatalog, name: &str, expected: &[usize]) -> Result<()> {
    let tensor = catalog.get(name).ok_or_else(|| invalid(name, "companion tensor is missing"))?;
    shape(name, &tensor.shape, expected)
}

pub(super) fn companion_dtype(catalog: &TensorCatalog, name: &str, expected: &str) -> Result<()> {
    let tensor = catalog.get(name).ok_or_else(|| invalid(name, "companion tensor is missing"))?;
    if tensor.dtype == expected {
        Ok(())
    } else {
        Err(invalid(name, &format!("expected dtype {expected}, found {}", tensor.dtype)))
    }
}

pub(super) fn shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid(name, &format!("expected shape {expected:?}, found {actual:?}")))
    }
}

pub(super) fn invalid(name: &str, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid tensor binding {name}: {reason}"))
}
