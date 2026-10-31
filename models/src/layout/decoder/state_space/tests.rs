use serde_json::{Value, json};

use crate::{
    error::{ModelsError, Result},
    layout::{AttentionLayerType, DecoderConfig, HybridLayer, StateSpaceStateDType, TimeStepBound},
};

/// The 88-block pattern of the Nemotron-3.5 Super checkpoint.
const SUPER_PATTERN: &str =
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME";

fn blocks(pattern: &str) -> Vec<&'static str> {
    pattern
        .chars()
        .map(|block| match block {
            'M' => "mamba",
            'E' => "moe",
            '*' => "attention",
            _ => "mlp",
        })
        .collect()
}

/// A multimodal configuration nesting the decoder under `llm_config`.
fn config(pattern: &str) -> Value {
    json!({
        "model_type": "nemotron_h_omni",
        "llm_config": {
            "model_type": "nemotron_h",
            "hidden_size": 4096,
            "intermediate_size": 2688,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 131_072,
            "layer_norm_epsilon": 1e-5,
            "max_position_embeddings": 1_048_576,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "layers_block_type": blocks(pattern),
            "mamba_num_heads": 128,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "n_groups": 8,
            "conv_kernel": 4,
            "use_conv_bias": true,
            "mamba_proj_bias": false,
            "chunk_size": 128,
            "expand": 2,
            "mamba_hidden_act": "silu",
            "mamba_ssm_cache_dtype": "float32",
            "time_step_limit": [0.0, { "__float__": "Infinity" }],
            "n_routed_experts": 512,
            "num_experts_per_tok": 22,
            "moe_latent_size": 1024,
            "moe_intermediate_size": 2688,
            "moe_shared_expert_intermediate_size": 5376,
            "n_shared_experts": 1,
            "routed_scaling_factor": 5.0,
            "norm_topk_prob": true,
            "n_group": 1,
            "topk_group": 1,
            "mlp_hidden_act": "relu2",
            "mlp_bias": false
        }
    })
}

fn with(mut config: Value, key: &str, value: Value) -> Value {
    config["llm_config"][key] = value;
    config
}

fn rejects(config: &Value, needle: &str) -> Result<()> {
    match DecoderConfig::from_value(config) {
        Ok(_) => Err(missing("configuration must be rejected")),
        Err(error) => {
            assert!(error.to_string().contains(needle), "unexpected error: {error}");
            Ok(())
        },
    }
}

fn missing(message: &str) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[test]
fn pairs_the_super_pattern_into_forty_eight_layers() -> Result<()> {
    let decoder = DecoderConfig::from_value(&config(SUPER_PATTERN))?;
    let hybrid = decoder.state_space.as_ref().ok_or_else(|| missing("hybrid configuration"))?;

    assert_eq!(hybrid.blocks.physical().len(), 88);
    assert_eq!(decoder.num_hidden_layers, 48);
    let count = |predicate: fn(&HybridLayer) -> bool| {
        hybrid.blocks.layers().iter().filter(|layer| predicate(layer)).count()
    };
    assert_eq!(count(|layer| matches!(layer, HybridLayer::MambaWithMoe { .. })), 32);
    assert_eq!(count(|layer| matches!(layer, HybridLayer::Mamba { .. })), 8);
    assert_eq!(count(|layer| matches!(layer, HybridLayer::AttentionWithMoe { .. })), 8);
    assert_eq!(hybrid.blocks.layers()[3], HybridLayer::Mamba { block: 6 });
    assert_eq!(hybrid.blocks.layers()[4], HybridLayer::AttentionWithMoe { block: 7 });
    let full = decoder
        .layer_types
        .iter()
        .filter(|kind| **kind == AttentionLayerType::Full)
        .count();
    assert_eq!(full, 8);
    assert_eq!(decoder.layer_type(0), AttentionLayerType::StateSpace);
    assert_eq!(decoder.num_experts, Some(512));
    Ok(())
}

#[test]
fn derives_mamba_and_latent_routed_geometry() -> Result<()> {
    let decoder = DecoderConfig::from_value(&config(SUPER_PATTERN))?;
    let hybrid = decoder.state_space.ok_or_else(|| missing("hybrid configuration"))?;

    assert_eq!(hybrid.mamba.inner_size(), 8192);
    assert_eq!(hybrid.mamba.convolution_channels(), 10_240);
    assert_eq!(hybrid.mamba.input_projection_size(), 18_560);
    assert_eq!(hybrid.mamba.norm_group_size(), 1024);
    assert_eq!(hybrid.mamba.state_dtype, StateSpaceStateDType::Float32);
    assert!(hybrid.mamba.time_step_limit.minimum.abs() < f64::EPSILON);
    assert_eq!(hybrid.mamba.time_step_limit.maximum, TimeStepBound::Unbounded);
    assert_eq!(hybrid.routed.latent_size, 1024);
    assert_eq!(hybrid.routed.shared_intermediate_size, Some(5376));
    assert!((hybrid.routed.routed_scale - 5.0).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn reads_a_finite_time_step_bound() -> Result<()> {
    let decoder =
        DecoderConfig::from_value(&with(config("ME"), "time_step_limit", json!([0.001, 0.1])))?;
    let limit = decoder
        .state_space
        .ok_or_else(|| missing("hybrid configuration"))?
        .mamba
        .time_step_limit;
    assert_eq!(limit.maximum, TimeStepBound::Finite(0.1));
    Ok(())
}

#[test]
fn rejects_malformed_block_patterns() -> Result<()> {
    rejects(&config("EM"), "does not follow a mixer")?;
    rejects(&config("MEE"), "does not follow a mixer")?;
    rejects(&config("ME*"), "not followed by a routed-expert block")?;
    rejects(&config("*M"), "not followed by a routed-expert block")?;
    rejects(&config("M-"), "unsupported hybrid block type mlp")?;
    rejects(&with(config("ME"), "layers_block_type", json!([])), "pattern is empty")?;
    rejects(&with(config("ME"), "layers_block_type", json!("ME")), "must be an array")?;
    rejects(&with(config("ME"), "num_hidden_layers", json!(3)), "num_hidden_layers is 3")?;
    Ok(())
}

#[test]
fn rejects_inconsistent_mamba_geometry() -> Result<()> {
    rejects(&with(config("ME"), "n_groups", json!(3)), "divisible by n_groups")?;
    rejects(&with(config("ME"), "expand", json!(4)), "expand * hidden_size")?;
    rejects(&with(config("ME"), "mamba_head_dim", json!(0)), "greater than zero")?;
    rejects(&with(config("ME"), "mamba_hidden_act", json!("gelu")), "silu")?;
    rejects(
        &with(config("ME"), "mamba_ssm_cache_dtype", json!("int8")),
        "mamba_ssm_cache_dtype",
    )?;
    rejects(&with(config("ME"), "time_step_limit", json!([0.5, 0.1])), "ordered")?;
    rejects(&with(config("ME"), "time_step_limit", json!([0.0])), "two entries")?;
    rejects(&with(config("ME"), "time_step_limit", json!([0.0, "big"])), "Infinity")?;
    Ok(())
}

#[test]
fn rejects_unsupported_routed_contracts() -> Result<()> {
    rejects(&with(config("ME"), "mlp_hidden_act", json!("silu")), "relu2")?;
    rejects(&with(config("ME"), "norm_topk_prob", json!(false)), "renormalized")?;
    rejects(&with(config("ME"), "n_group", json!(8)), "grouped expert selection")?;
    rejects(&with(config("ME"), "num_experts_per_tok", json!(513)), "cannot exceed")?;
    rejects(&with(config("ME"), "n_shared_experts", json!(2)), "more than one shared expert")?;
    rejects(&with(config("ME"), "routed_scaling_factor", json!(0.0)), "finite and positive")?;
    rejects(&with(config("ME"), "mlp_bias", json!(true)), "projection bias")?;
    Ok(())
}
