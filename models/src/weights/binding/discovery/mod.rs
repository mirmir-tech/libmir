use std::collections::BTreeSet;

use super::{
    AffineGroupAxis, AffinePacking, AffineParameterDType, AffineSignedness, AffineStorageDType,
    AffineZeroPointMode, BlockQuantization, GroupedAffineQuantization, LayerTensorRole,
    LogicalTensorRole, TensorBinding, TensorStorage, WeightBindingPlan, dimensions, grammar, roles,
    shapes, validation,
};
use crate::{
    error::{ModelsError, Result},
    layout::{ModelMetadata, VisionConfig},
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, TensorInfo},
};

mod awq;
mod bitsandbytes;
mod block;
mod companions;
mod float8;
mod gptq;
mod packed_integer;
#[cfg(test)]
mod tests;

use companions::{companion, is_companion};

pub(super) fn discover(
    spec: &SemanticModelSpec,
    catalog: &TensorCatalog,
) -> Result<WeightBindingPlan> {
    discover_with_hints(spec, catalog, DiscoveryHints::default())
}

pub(super) fn discover_from_layout(
    spec: &SemanticModelSpec,
    catalog: &TensorCatalog,
    layout: &crate::layout::ModelLayout,
) -> Result<WeightBindingPlan> {
    let hints = DiscoveryHints {
        bitsandbytes: bitsandbytes::hint(layout)?,
        gptq: gptq::hint(layout)?,
        float8: float8::hint(layout)?,
        nvfp4: block::hint(layout)?,
        block: match ModelMetadata::from_layout(layout)?.quantization {
            foundation::model::Quantization::MxFp4 => Some(BlockQuantization::MXFP4_MLX),
            foundation::model::Quantization::MxFp8 => Some(BlockQuantization::MXFP8),
            _ => None,
        },
        vision_projection: match VisionConfig::from_layout(layout)? {
            Some(VisionConfig::PooledEncoder(config)) => {
                Some([config.output_hidden_size, config.hidden_size])
            },
            Some(VisionConfig::SpatialMergeEncoder(_)) | None => None,
        },
    };
    discover_with_hints(spec, catalog, hints)
}

#[derive(Clone, Copy, Default)]
struct DiscoveryHints {
    bitsandbytes: Option<bitsandbytes::Hint>,
    gptq: Option<gptq::GptqHint>,
    float8: Option<float8::Float8Hint>,
    nvfp4: Option<BlockQuantization>,
    block: Option<BlockQuantization>,
    vision_projection: Option<[usize; 2]>,
}

fn discover_with_hints(
    spec: &SemanticModelSpec,
    catalog: &TensorCatalog,
    hints: DiscoveryHints,
) -> Result<WeightBindingPlan> {
    validate_layer_coverage(spec, catalog)?;
    let mut consumed = BTreeSet::new();
    let mut tensors = Vec::new();
    for tensor in &catalog.tensors {
        if consumed.contains(&tensor.name) || is_companion(&tensor.name, catalog) {
            continue;
        }
        let binding = bind(spec, tensor, catalog, &mut consumed, hints)?;
        tensors.push(binding);
    }
    tensors.sort_by(|left, right| left.role.cmp(&right.role));
    grammar::validate(spec, &tensors)?;
    let plan = WeightBindingPlan { tensors };
    validation::validate(&plan, catalog)?;
    Ok(plan)
}

fn validate_layer_coverage(spec: &SemanticModelSpec, catalog: &TensorCatalog) -> Result<()> {
    for layer in &spec.decoder.layers {
        let needle = format!("layers.{}.", layer.index);
        if !catalog.tensors.iter().any(|tensor| tensor.name.contains(&needle)) {
            return Err(ModelsError::InvalidConfig(format!(
                "checkpoint has no tensors for semantic layer {}",
                layer.index
            )));
        }
    }
    Ok(())
}

fn bind(
    spec: &SemanticModelSpec,
    tensor: &TensorInfo,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
    hints: DiscoveryHints,
) -> Result<TensorBinding> {
    let _inserted = consumed.insert(tensor.name.clone());
    let role = roles::parse(&tensor.name);
    let logical_shape = shapes::logical(spec, &role, hints.vision_projection);
    let storage = if let Some(prefix) = tensor.name.strip_suffix("_blocks") {
        block::mxfp4_storage(prefix, catalog, consumed)
    } else if let Some(prefix) = tensor.name.strip_suffix(".qweight") {
        match hints.gptq {
            Some(hint) => {
                gptq::storage(prefix, logical_shape.as_deref(), tensor, catalog, consumed, hint)?
            },
            None => awq::storage(prefix, logical_shape.as_deref(), tensor, catalog, consumed)?,
        }
    } else if let Some(prefix) = tensor.name.strip_suffix(".weight_packed") {
        match block::nvfp4_packed_storage(prefix, catalog, consumed, hints.nvfp4) {
            Some(storage) => storage,
            None => packed_integer::storage(
                prefix,
                logical_shape.as_deref(),
                tensor,
                catalog,
                consumed,
            )?,
        }
    } else if tensor.name.ends_with(".weight")
        || matches!(
            role,
            LogicalTensorRole::Layer {
                tensor: LayerTensorRole::ExpertProjection { .. },
                ..
            }
        )
    {
        projection_storage(logical_shape.as_deref(), tensor, catalog, consumed, hints)?
    } else {
        TensorStorage::Auxiliary { dtype: tensor.dtype.clone() }
    };
    let transforms = shapes::transforms(
        spec,
        &role,
        &tensor.name,
        &tensor.shape,
        logical_shape.as_deref(),
        &storage,
    );
    Ok(TensorBinding {
        role,
        source: tensor.name.clone(),
        shape: tensor.shape.clone(),
        logical_shape,
        transforms,
        storage,
    })
}

fn projection_storage(
    logical_shape: Option<&[usize]>,
    tensor: &TensorInfo,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
    hints: DiscoveryHints,
) -> Result<TensorStorage> {
    if let Some(storage) = bitsandbytes::storage(tensor, catalog, consumed, hints.bitsandbytes)? {
        return Ok(storage);
    }
    let prefix = tensor.name.trim_end_matches(".weight");
    if let Some(storage) = block::mlx_storage(prefix, catalog, consumed, hints.block) {
        return Ok(storage);
    }
    if let Some(storage) = float8::storage(prefix, tensor, catalog, consumed, hints.float8)? {
        return Ok(storage);
    }
    if let Some(storage) = block::nvfp4_storage(prefix, catalog, consumed, hints.nvfp4) {
        return Ok(storage);
    }
    let scales = format!("{prefix}.scales");
    let biases = catalog
        .contains(&format!("{prefix}.biases"))
        .then(|| format!("{prefix}.biases"));
    let output_bias = companion(catalog, [format!("{prefix}.bias"), format!("{prefix}_bias")]);
    if catalog.contains(&scales) {
        let _inserted = consumed.insert(scales.clone());
        if let Some(name) = &biases {
            let _inserted = consumed.insert(name.clone());
        }
        if let Some(name) = &output_bias {
            let _inserted = consumed.insert(name.clone());
        }
        let scales_tensor = catalog
            .get(&scales)
            .ok_or_else(|| ModelsError::InvalidConfig(format!("missing affine scales {scales}")))?;
        let (bits, group_size) =
            dimensions::affine_geometry(&tensor.name, logical_shape, tensor, scales_tensor)?;
        let scale_dtype = AffineParameterDType::parse(scales_tensor)?;
        let bias_dtype = biases
            .as_deref()
            .map(|name| {
                catalog
                    .get(name)
                    .ok_or_else(|| {
                        ModelsError::InvalidConfig(format!("missing affine biases {name}"))
                    })
                    .and_then(AffineParameterDType::parse)
            })
            .transpose()?;
        Ok(TensorStorage::AffineQuantized {
            format: GroupedAffineQuantization {
                bits,
                group_size,
                group_axis: AffineGroupAxis::Input,
                signedness: AffineSignedness::Unsigned,
                zero_point: if biases.is_some() {
                    AffineZeroPointMode::AdditiveBias
                } else {
                    AffineZeroPointMode::None
                },
                packing: AffinePacking::Mlx,
                storage_dtype: AffineStorageDType::parse(tensor)?,
                scale_dtype,
                bias_dtype,
            },
            scales,
            biases,
            output_bias,
        })
    } else {
        if let Some(name) = &output_bias {
            let _inserted = consumed.insert(name.clone());
        }
        Ok(TensorStorage::Dense {
            dtype: tensor.dtype.clone(),
            bias: output_bias,
        })
    }
}
