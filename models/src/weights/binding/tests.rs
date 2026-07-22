#![allow(clippy::self_named_module_files)]

use std::path::PathBuf;

use serde_json::json;

use super::*;
use crate::{
    error::Result,
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, TensorInfo},
};

mod dense;
mod hybrid;
mod mistral;
mod packed_int8;
mod roles;
mod view;

#[test]
fn physical_checkpoint_formats_do_not_change_model_semantics() -> Result<()> {
    let decoder = decoder()?;
    let native = native_catalog();
    let mlx = mlx_catalog();
    let native_spec = SemanticModelSpec::discover(&decoder, &native)?;
    let mlx_spec = SemanticModelSpec::discover(&decoder, &mlx)?;

    assert_eq!(native_spec, mlx_spec);
    let native_bindings = WeightBindingPlan::discover(&native_spec, &native)?;
    let mlx_bindings = WeightBindingPlan::discover(&mlx_spec, &mlx)?;
    assert!(
        native_bindings
            .tensors
            .iter()
            .any(|binding| { matches!(binding.storage, TensorStorage::BlockQuantized { .. }) })
    );
    assert!(
        mlx_bindings
            .tensors
            .iter()
            .any(|binding| { matches!(binding.storage, TensorStorage::AffineQuantized { .. }) })
    );
    assert_eq!(
        native_bindings.expert_projection_layout(),
        Some(ExpertProjectionLayout::InterleavedGateUp)
    );
    assert_eq!(
        mlx_bindings.expert_projection_layout(),
        Some(ExpertProjectionLayout::SeparateGateUp)
    );
    assert_ne!(native_bindings, mlx_bindings);
    Ok(())
}

#[test]
fn rejects_ambiguous_expert_binding_grammars() -> Result<()> {
    let decoder = decoder()?;
    let mut ambiguous = native_catalog();
    ambiguous.tensors.extend(mlx_catalog().tensors);
    let spec = SemanticModelSpec::discover(&decoder, &ambiguous)?;

    let error = WeightBindingPlan::discover(&spec, &ambiguous);

    assert!(error.is_err());
    Ok(())
}

#[test]
fn attaches_nvfp4_storage_to_each_physical_binding() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "hidden_act": "silu"
    }))?;
    let mut catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight", "U8", vec![32, 16]),
            tensor("model.layers.0.self_attn.q_proj.weight_scale", "F8_E4M3", vec![32, 2]),
            tensor("model.layers.0.self_attn.q_proj.weight_scale_2", "F32", Vec::new()),
            tensor("model.layers.0.self_attn.q_proj.input_scale", "F32", Vec::new()),
        ],
    };
    catalog.tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;

    assert!(bindings.uses_block_format(BlockFormat::NvFp4));
    assert_eq!(bindings.tensors.len(), 1);
    assert!(matches!(
        bindings.tensors[0].storage,
        TensorStorage::BlockQuantized {
            format: BlockFormat::NvFp4,
            global_scale: Some(_),
            input_scale: Some(_),
            ..
        }
    ));
    Ok(())
}

#[test]
fn derives_affine_bits_and_group_size_from_semantic_dimensions() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "hidden_act": "silu"
    }))?;
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight", "U32", vec![32, 4]),
            tensor("model.layers.0.self_attn.q_proj.scales", "F16", vec![32, 2]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;

    assert_eq!(bindings.affine_group_size(), Some(16));
    assert!(matches!(
        bindings.tensors[0].storage,
        TensorStorage::AffineQuantized { bits: Some(4), group_size: Some(16), .. }
    ));
    Ok(())
}

fn decoder() -> Result<DecoderConfig> {
    DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "hidden_act": "silu",
        "attention_bias": true,
        "swiglu_limit": 7.0,
        "layer_types": ["full_attention"]
    }))
}

fn native_catalog() -> TensorCatalog {
    TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.sinks", "F32", vec![4]),
            tensor("model.layers.0.mlp.experts.gate_up_proj_blocks", "U8", vec![8, 64, 1, 16]),
            tensor("model.layers.0.mlp.experts.gate_up_proj_scales", "F8_E4M3", vec![8, 64, 1]),
            tensor("model.layers.0.mlp.experts.down_proj_blocks", "U8", vec![8, 32, 1, 16]),
            tensor("model.layers.0.mlp.experts.down_proj_scales", "F8_E4M3", vec![8, 32, 1]),
        ],
    }
}

fn mlx_catalog() -> TensorCatalog {
    TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.sinks", "F32", vec![4]),
            tensor("model.layers.0.mlp.experts.gate_proj.weight", "U32", vec![8, 32, 4]),
            tensor("model.layers.0.mlp.experts.gate_proj.scales", "F16", vec![8, 32, 1]),
            tensor("model.layers.0.mlp.experts.up_proj.weight", "U32", vec![8, 32, 4]),
            tensor("model.layers.0.mlp.experts.up_proj.scales", "F16", vec![8, 32, 1]),
            tensor("model.layers.0.mlp.experts.down_proj.weight", "U32", vec![8, 32, 4]),
            tensor("model.layers.0.mlp.experts.down_proj.scales", "F16", vec![8, 32, 1]),
        ],
    }
}

fn tensor(name: &str, dtype: &str, shape: Vec<usize>) -> TensorInfo {
    TensorInfo {
        name: name.to_owned(),
        file: PathBuf::new(),
        dtype: dtype.to_owned(),
        shape,
        data_start: 0,
        data_offsets: [0, 0],
    }
}
