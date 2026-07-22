use std::collections::BTreeSet;

use super::{
    BlockFormat, TensorBinding, TensorPacking, TensorStorage, WeightBindingPlan, dimensions,
    grammar, roles, shapes, validation,
};
use crate::{
    error::{ModelsError, Result},
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, TensorInfo},
};

pub(super) fn discover(
    spec: &SemanticModelSpec,
    catalog: &TensorCatalog,
) -> Result<WeightBindingPlan> {
    validate_layer_coverage(spec, catalog)?;
    let mut consumed = BTreeSet::new();
    let mut tensors = Vec::new();
    for tensor in &catalog.tensors {
        if consumed.contains(&tensor.name) || is_companion(&tensor.name, catalog) {
            continue;
        }
        let binding = bind(spec, tensor, catalog, &mut consumed)?;
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
) -> Result<TensorBinding> {
    let _inserted = consumed.insert(tensor.name.clone());
    let role = roles::parse(&tensor.name);
    let logical_shape = shapes::logical(spec, &role);
    let storage = if let Some(prefix) = tensor.name.strip_suffix("_blocks") {
        block_storage(prefix, catalog, consumed)
    } else if let Some(prefix) = tensor.name.strip_suffix(".weight_packed") {
        packed_int8_storage(prefix, tensor, catalog, consumed)?
    } else if tensor.name.ends_with(".weight") {
        projection_storage(logical_shape.as_deref(), tensor, catalog, consumed)?
    } else {
        TensorStorage::Auxiliary { dtype: tensor.dtype.clone() }
    };
    let transforms = shapes::transforms(spec, &role, &tensor.shape, &storage);
    Ok(TensorBinding {
        role,
        source: tensor.name.clone(),
        shape: tensor.shape.clone(),
        logical_shape,
        transforms,
        storage,
    })
}

fn packed_int8_storage(
    prefix: &str,
    tensor: &TensorInfo,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
) -> Result<TensorStorage> {
    let scales = format!("{prefix}.weight_scale");
    if !catalog.contains(&scales) {
        return Err(ModelsError::InvalidConfig(format!("missing packed INT8 scales {scales}")));
    }
    let _inserted = consumed.insert(scales.clone());
    let _inserted = consumed.insert(format!("{prefix}.weight_shape"));
    Ok(TensorStorage::PackedInt8 { dtype: tensor.dtype.clone(), scales })
}

fn block_storage(
    prefix: &str,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
) -> TensorStorage {
    let scales = format!("{prefix}_scales");
    let bias = companion(catalog, [format!("{prefix}_bias"), format!("{prefix}.bias")]);
    let _inserted = consumed.insert(scales.clone());
    if let Some(name) = &bias {
        let _inserted = consumed.insert(name.clone());
    }
    TensorStorage::BlockQuantized {
        format: BlockFormat::MxFp4,
        scales,
        global_scale: None,
        input_scale: None,
        bias,
        packing: if prefix.ends_with("gate_up_proj") {
            TensorPacking::InterleavedGateUp
        } else {
            TensorPacking::Separate
        },
    }
}

fn projection_storage(
    logical_shape: Option<&[usize]>,
    tensor: &TensorInfo,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
) -> Result<TensorStorage> {
    let prefix = tensor.name.trim_end_matches(".weight");
    let block_scales = format!("{prefix}.weight_scale");
    let global_scale = format!("{prefix}.weight_scale_2");
    let input_scale = format!("{prefix}.input_scale");
    if catalog.contains(&block_scales)
        && catalog.contains(&global_scale)
        && catalog.contains(&input_scale)
    {
        let _inserted = consumed.insert(block_scales.clone());
        let _inserted = consumed.insert(global_scale.clone());
        let _inserted = consumed.insert(input_scale.clone());
        return Ok(TensorStorage::BlockQuantized {
            format: BlockFormat::NvFp4,
            scales: block_scales,
            global_scale: Some(global_scale),
            input_scale: Some(input_scale),
            bias: None,
            packing: TensorPacking::Separate,
        });
    }
    let scales = format!("{prefix}.scales");
    let biases = catalog
        .contains(&format!("{prefix}.biases"))
        .then(|| format!("{prefix}.biases"));
    let output_bias = catalog.contains(&format!("{prefix}.bias")).then(|| format!("{prefix}.bias"));
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
        Ok(TensorStorage::AffineQuantized {
            dtype: tensor.dtype.clone(),
            bits,
            scales,
            biases,
            output_bias,
            group_size,
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

fn companion(catalog: &TensorCatalog, candidates: [String; 2]) -> Option<String> {
    candidates.into_iter().find(|name| catalog.contains(name))
}

#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn is_companion(name: &str, catalog: &TensorCatalog) -> bool {
    name.ends_with(".scales")
        || name.ends_with(".biases")
        || name.ends_with("_scales")
        || name.ends_with(".weight_scale")
        || name.ends_with(".weight_scale_2")
        || name.ends_with(".input_scale")
        || name.ends_with(".weight_shape")
        || name.ends_with(".bias")
        || name
            .strip_suffix("_bias")
            .is_some_and(|prefix| catalog.contains(&format!("{prefix}_blocks")))
}
