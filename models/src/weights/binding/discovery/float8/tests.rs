use std::{collections::BTreeSet, fs, path::PathBuf};

use super::*;
use crate::{layout::ModelLayout, weights::TensorInfo};

#[test]
fn reads_compressed_tensors_dynamic_token_activation_contract() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-fp8-hint-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "input_activations": {
                            "dynamic": true,
                            "num_bits": 8,
                            "strategy": "token",
                            "type": "float"
                        },
                        "weights": {
                            "dynamic": false,
                            "num_bits": 8,
                            "strategy": "channel",
                            "type": "float"
                        }
                    }
                }
            }
        }))?,
    )?;
    fs::write(root.join("model.safetensors"), [])?;
    let layout = ModelLayout::inspect(&root)?;
    let parsed = hint(&layout)?.ok_or_else(|| invalid("config", "missing FP8 hint"))?;
    assert_eq!(parsed.activation_scale, Float8ActivationScale::DynamicToken);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn retains_declared_dimensions_for_a_padded_block_grid() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-fp8-block-hint-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "dynamic": false,
                            "num_bits": 8,
                            "strategy": "block",
                            "block_structure": [4, 8],
                            "type": "float"
                        }
                    }
                }
            }
        }))?,
    )?;
    fs::write(root.join("model.safetensors"), [])?;
    let layout = ModelLayout::inspect(&root)?;
    let parsed = hint(&layout)?.ok_or_else(|| invalid("config", "missing FP8 hint"))?;
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("projection.weight", "F8_E4M3", vec![5, 12]),
            tensor("projection.weight_scale", "F32", vec![2, 2]),
        ],
    };
    let storage =
        storage("projection", &catalog.tensors[0], &catalog, &mut BTreeSet::new(), Some(parsed))?
            .ok_or_else(|| invalid("projection", "float8 storage was not discovered"))?;
    let TensorStorage::Float8 { format, .. } = storage else {
        return Err(invalid("projection", "storage is not float8"));
    };
    assert_eq!(
        format.scale_granularity,
        Float8ScaleGranularity::BlockGrid {
            output_groups: 2,
            input_groups: 2,
            output_block_size: Some(4),
            input_block_size: Some(8),
        }
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn discovers_e4m3_multiplier_and_e5m2_inverse_scale() -> Result<()> {
    for (dtype, suffix, expected_format, expected_mode) in [
        ("F8_E4M3", "weight_scale", Float8Format::E4M3, Float8ScaleMode::Multiplier),
        (
            "F8_E5M2",
            "weight_scale_inv",
            Float8Format::E5M2,
            Float8ScaleMode::InverseMultiplier,
        ),
    ] {
        let catalog = catalog(dtype, suffix, Vec::new());
        let storage =
            storage("projection", &catalog.tensors[0], &catalog, &mut BTreeSet::new(), None)?
                .ok_or_else(|| invalid("projection", "float8 storage was not discovered"))?;
        let TensorStorage::Float8 { format, .. } = storage else {
            return Err(invalid("projection", "storage is not float8"));
        };
        assert_eq!(format.format, expected_format);
        assert_eq!(format.scale_mode, expected_mode);
    }
    Ok(())
}

#[test]
fn discovers_output_channel_and_block_grid_scales() -> Result<()> {
    for (scale_shape, expected) in [
        (vec![4], Float8ScaleGranularity::OutputChannel),
        (vec![4, 1], Float8ScaleGranularity::OutputChannel),
        (
            vec![2, 2],
            Float8ScaleGranularity::BlockGrid {
                output_groups: 2,
                input_groups: 2,
                output_block_size: None,
                input_block_size: None,
            },
        ),
    ] {
        let catalog = catalog("F8_E4M3", "weight_scale", scale_shape);
        let storage =
            storage("projection", &catalog.tensors[0], &catalog, &mut BTreeSet::new(), None)?
                .ok_or_else(|| invalid("projection", "float8 storage was not discovered"))?;
        let TensorStorage::Float8 { format, .. } = storage else {
            return Err(invalid("projection", "storage is not float8"));
        };
        assert_eq!(format.scale_granularity, expected);
    }
    Ok(())
}

#[test]
fn discovers_singleton_tensor_scale() -> Result<()> {
    let catalog = catalog("F8_E4M3", "weight_scale", vec![1]);
    let storage = storage("projection", &catalog.tensors[0], &catalog, &mut BTreeSet::new(), None)?
        .ok_or_else(|| invalid("projection", "float8 storage was not discovered"))?;
    let TensorStorage::Float8 { format, .. } = storage else {
        return Err(invalid("projection", "storage is not float8"));
    };
    assert_eq!(format.scale_granularity, Float8ScaleGranularity::Tensor);
    Ok(())
}

#[test]
fn rejects_ambiguous_or_invalid_scale_contracts() {
    let mut ambiguous = catalog("F8_E4M3", "weight_scale", Vec::new());
    ambiguous.tensors.push(tensor("projection.weight_scale_inv", "F32", Vec::new()));
    assert!(
        storage("projection", &ambiguous.tensors[0], &ambiguous, &mut BTreeSet::new(), None,)
            .is_err()
    );

    let catalog = catalog("F8_E4M3", "weight_scale", vec![3]);
    assert!(
        storage("projection", &catalog.tensors[0], &catalog, &mut BTreeSet::new(), None,).is_err()
    );
}

fn catalog(dtype: &str, scale_suffix: &str, scale_shape: Vec<usize>) -> TensorCatalog {
    TensorCatalog {
        tensors: vec![
            tensor("projection.weight", dtype, vec![4, 8]),
            tensor(&format!("projection.{scale_suffix}"), "F32", scale_shape),
        ],
    }
}

fn tensor(name: &str, dtype: &str, shape: Vec<usize>) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: PathBuf::new(),
        dtype: dtype.into(),
        shape,
        data_start: 0,
        data_offsets: [0, 0],
    }
}
