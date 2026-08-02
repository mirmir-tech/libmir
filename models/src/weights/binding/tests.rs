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

mod affine;
mod awq;
mod block;
mod dense;
mod gptq;
mod hybrid;
mod mistral;
mod packed_integer;
mod roles;
mod view;
mod vision;

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
            format: BlockQuantization::NVFP4,
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
    let catalog = TensorCatalog::new(vec![
        tensor("model.layers.0.self_attn.q_proj.weight", "U32", vec![32, 4]),
        tensor("model.layers.0.self_attn.q_proj.scales", "F16", vec![32, 2]),
    ]);
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;

    assert_eq!(bindings.affine_group_size(), Some(16));
    assert!(matches!(
        bindings.tensors[0].storage,
        TensorStorage::AffineQuantized {
            format: GroupedAffineQuantization {
                bits: AffineBits::Four,
                group_size: 16,
                ..
            },
            ..
        }
    ));
    Ok(())
}

#[test]
fn binds_transposed_dense_gpt_oss_experts_and_suffix_biases() -> Result<()> {
    let decoder = decoder()?;
    let catalog = TensorCatalog::new(vec![
        tensor("model.layers.0.self_attn.sinks", "F32", vec![4]),
        tensor("model.layers.0.mlp.experts.gate_up_proj", "BF16", vec![8, 32, 64]),
        tensor("model.layers.0.mlp.experts.gate_up_proj_bias", "BF16", vec![8, 64]),
        tensor("model.layers.0.mlp.experts.down_proj", "BF16", vec![8, 32, 32]),
        tensor("model.layers.0.mlp.experts.down_proj_bias", "BF16", vec![8, 32]),
    ]);
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;
    let expert = |projection| LogicalTensorRole::Layer {
        index: 0,
        tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
    };
    let gate_up = bindings
        .binding(&expert(ExpertProjectionRole::GateUp))
        .ok_or_else(|| crate::ModelsError::InvalidConfig("missing fused dense expert".into()))?;
    let down = bindings
        .binding(&expert(ExpertProjectionRole::Down))
        .ok_or_else(|| crate::ModelsError::InvalidConfig("missing dense down expert".into()))?;
    assert_eq!(
        gate_up.storage,
        TensorStorage::Dense {
            dtype: "BF16".into(),
            bias: Some("model.layers.0.mlp.experts.gate_up_proj_bias".into()),
        }
    );
    assert!(gate_up.transforms.contains(&BindingTransform::Transpose));
    assert!(
        gate_up
            .transforms
            .contains(&BindingTransform::FusedGateUp { interleaved: true })
    );
    assert!(down.transforms.contains(&BindingTransform::Transpose));
    assert_eq!(down.physical_sources().len(), 2);
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
