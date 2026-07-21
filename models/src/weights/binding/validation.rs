use std::collections::BTreeMap;

use super::{
    BindingTransform, BlockFormat, LogicalTensorRole, TensorBinding, TensorStorage,
    WeightBindingPlan,
};
use crate::{
    error::{ModelsError, Result},
    weights::TensorCatalog,
};

pub(super) fn validate(plan: &WeightBindingPlan, catalog: &TensorCatalog) -> Result<()> {
    unique_roles(plan)?;
    for binding in &plan.tensors {
        let Some(logical) = binding.logical_shape.as_deref() else {
            continue;
        };
        match &binding.storage {
            TensorStorage::Dense { bias, .. } => {
                let expected = dense_shape(binding, logical);
                shape(&binding.source, &binding.shape, &expected)?;
                if let Some(bias) = bias
                    && logical.len() > 1
                {
                    companion(catalog, bias, &logical[..logical.len() - 1])?;
                }
            },
            TensorStorage::AffineQuantized {
                bits,
                scales,
                biases,
                output_bias,
                group_size,
                ..
            } => {
                affine(binding, logical, *bits, *group_size, scales, biases.as_deref(), catalog)?;
                if let Some(output_bias) = output_bias
                    && logical.len() > 1
                {
                    companion(catalog, output_bias, &logical[..logical.len() - 1])?;
                }
            },
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

fn dense_shape(binding: &TensorBinding, logical: &[usize]) -> Vec<usize> {
    if binding.transforms.contains(&BindingTransform::Transpose) {
        logical.iter().rev().copied().collect()
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

fn affine(
    binding: &TensorBinding,
    logical: &[usize],
    bits: Option<u8>,
    group_size: Option<usize>,
    scales: &str,
    biases: Option<&str>,
    catalog: &TensorCatalog,
) -> Result<()> {
    let bits = usize::from(bits.ok_or_else(|| invalid(&binding.source, "missing bit width"))?);
    let group = group_size.ok_or_else(|| invalid(&binding.source, "missing group size"))?;
    let (prefix, input) = projection(logical, &binding.source)?;
    let packed = input
        .checked_mul(bits)
        .filter(|value| value.is_multiple_of(32))
        .map(|value| value / 32)
        .ok_or_else(|| invalid(&binding.source, "logical input cannot be packed into U32"))?;
    let mut physical = prefix.to_vec();
    physical.push(packed);
    shape(&binding.source, &binding.shape, &physical)?;
    let groups = input
        .checked_div(group)
        .filter(|groups| input.is_multiple_of(group) && *groups > 0)
        .ok_or_else(|| invalid(&binding.source, "invalid affine group geometry"))?;
    let mut parameters = prefix.to_vec();
    parameters.push(groups);
    companion(catalog, scales, &parameters)?;
    if let Some(biases) = biases {
        companion(catalog, biases, &parameters)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn block(
    binding: &TensorBinding,
    logical: &[usize],
    format: BlockFormat,
    scales: &str,
    global_scale: Option<&str>,
    input_scale: Option<&str>,
    bias: Option<&str>,
    catalog: &TensorCatalog,
) -> Result<()> {
    let (prefix, input) = projection(logical, &binding.source)?;
    let (divisor, tail) = match format {
        BlockFormat::MxFp4 => (32, Some(16)),
        BlockFormat::NvFp4 => (2, None),
    };
    let columns = divided(input, divisor, &binding.source)?;
    let mut physical = prefix.to_vec();
    physical.push(columns);
    if let Some(tail) = tail {
        physical.push(tail);
    }
    shape(&binding.source, &binding.shape, &physical)?;
    let scale_divisor = match format {
        BlockFormat::MxFp4 => 32,
        BlockFormat::NvFp4 => 16,
    };
    let mut scale_shape = prefix.to_vec();
    scale_shape.push(divided(input, scale_divisor, &binding.source)?);
    companion(catalog, scales, &scale_shape)?;
    if let Some(global) = global_scale {
        companion(catalog, global, &[])?;
    }
    if let Some(input_scale) = input_scale {
        companion(catalog, input_scale, &[])?;
    }
    if let Some(bias) = bias {
        companion(catalog, bias, prefix)?;
    }
    Ok(())
}

fn projection<'a>(logical: &'a [usize], source: &str) -> Result<(&'a [usize], usize)> {
    logical
        .split_last()
        .map(|(input, prefix)| (prefix, *input))
        .ok_or_else(|| invalid(source, "projection shape is empty"))
}

fn divided(value: usize, divisor: usize, source: &str) -> Result<usize> {
    value
        .checked_div(divisor)
        .filter(|result| value.is_multiple_of(divisor) && *result > 0)
        .ok_or_else(|| invalid(source, "block geometry does not divide logical input"))
}

fn companion(catalog: &TensorCatalog, name: &str, expected: &[usize]) -> Result<()> {
    let tensor = catalog.get(name).ok_or_else(|| invalid(name, "companion tensor is missing"))?;
    shape(name, &tensor.shape, expected)
}

fn shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid(name, &format!("expected shape {expected:?}, found {actual:?}")))
    }
}

fn invalid(name: &str, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid tensor binding {name}: {reason}"))
}
