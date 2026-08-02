use std::{collections::BTreeSet, fs};

use serde_json::Value;

use super::{BlockQuantization, TensorCatalog, TensorStorage, companion};
use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
    weights::{BlockFormat, TensorPacking},
};

pub(super) fn hint(layout: &ModelLayout) -> Result<Option<BlockQuantization>> {
    let root: Value = serde_json::from_str(&fs::read_to_string(&layout.config_path)?)?;
    hint_from(&root)
}

fn hint_from(root: &Value) -> Result<Option<BlockQuantization>> {
    let layers = root
        .get("quantization_config")
        .or_else(|| root.get("quantization"))
        .and_then(|config| config.get("quantized_layers"))
        .and_then(Value::as_object);
    let mut mode = None;
    for algorithm in layers
        .into_iter()
        .flatten()
        .filter_map(|(_, layer)| layer.get("quant_algo").and_then(Value::as_str))
        .filter(|algorithm| algorithm.to_ascii_uppercase().contains("NVFP4"))
    {
        let format = if algorithm.eq_ignore_ascii_case("W4A16_NVFP4") {
            BlockQuantization::NVFP4_W4A16
        } else if algorithm.eq_ignore_ascii_case("NVFP4") {
            BlockQuantization::NVFP4
        } else {
            return Err(invalid(format!("unsupported NVFP4 algorithm {algorithm}")));
        };
        if mode.replace(format).is_some_and(|previous| previous != format) {
            return Err(invalid("mixed NVFP4 activation contracts"));
        }
    }
    Ok(mode)
}

pub(super) fn nvfp4_storage(
    prefix: &str,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
    hint: Option<BlockQuantization>,
) -> Option<TensorStorage> {
    let scales = format!("{prefix}.weight_scale");
    let global_scale = format!("{prefix}.weight_scale_2");
    let input_scale = format!("{prefix}.input_scale");
    if !catalog.contains(&scales)
        || !catalog.contains(&global_scale)
        || !catalog.contains(&input_scale)
    {
        return None;
    }
    consumed.extend([scales.clone(), global_scale.clone(), input_scale.clone()]);
    Some(TensorStorage::BlockQuantized {
        format: hint.unwrap_or(BlockQuantization::NVFP4),
        scales,
        global_scale: Some(global_scale),
        input_scale: Some(input_scale),
        bias: None,
        packing: TensorPacking::Separate,
    })
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(format!("quantization_config: {}", message.into()))
}

pub(super) fn mxfp4_storage(
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
        format: BlockQuantization::MXFP4,
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

pub(super) fn mlx_storage(
    prefix: &str,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
    format: Option<BlockQuantization>,
) -> Option<TensorStorage> {
    let format =
        format.filter(|format| matches!(format.format, BlockFormat::MxFp4 | BlockFormat::MxFp8))?;
    let scales = format!("{prefix}.scales");
    if !catalog.contains(&scales) || catalog.contains(&format!("{prefix}.biases")) {
        return None;
    }
    let bias = companion(catalog, [format!("{prefix}.bias"), format!("{prefix}_bias")]);
    let _inserted = consumed.insert(scales.clone());
    if let Some(name) = &bias {
        let _inserted = consumed.insert(name.clone());
    }
    Some(TensorStorage::BlockQuantized {
        format,
        scales,
        global_scale: None,
        input_scale: None,
        bias,
        packing: TensorPacking::Separate,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn discovers_modelopt_weight_only_nvfp4() -> Result<()> {
        let root = json!({
            "quantization_config": {
                "quantized_layers": {
                    "model.layers.0.attn.q_proj": { "quant_algo": "FP8" },
                    "model.layers.0.mlp.experts": { "quant_algo": "W4A16_NVFP4" }
                }
            }
        });
        assert_eq!(hint_from(&root)?, Some(BlockQuantization::NVFP4_W4A16));
        Ok(())
    }

    #[test]
    fn rejects_mixed_nvfp4_activation_contracts() {
        let root = json!({
            "quantization_config": {
                "quantized_layers": {
                    "model.layers.0.mlp": { "quant_algo": "NVFP4" },
                    "model.layers.1.mlp": { "quant_algo": "W4A16_NVFP4" }
                }
            }
        });
        assert!(hint_from(&root).is_err());
    }
}
