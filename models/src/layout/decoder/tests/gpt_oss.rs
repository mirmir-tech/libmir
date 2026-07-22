use serde_json::{Value, json};

use super::super::{AttentionLayerType, DecoderConfig, RopeScaling};
use crate::error::{ModelsError, Result};

#[test]
fn reads_official_20b_geometry() -> Result<()> {
    let config = DecoderConfig::from_value(&configuration(24, 32))?;

    assert_contract(&config, 24, 32)
}

#[test]
fn reads_official_120b_geometry() -> Result<()> {
    let config = DecoderConfig::from_value(&configuration(36, 128))?;

    assert_contract(&config, 36, 128)
}

fn configuration(layers: usize, experts: usize) -> Value {
    let mut value = json!({
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": layers,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 201_088,
        "max_position_embeddings": 131_072,
        "num_local_experts": experts,
        "experts_per_token": 4,
        "sliding_window": 128,
        "hidden_act": "silu",
        "attention_bias": true,
        "swiglu_limit": 7.0,
        "rope_theta": 150_000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096
        }
    });
    value["layer_types"] = Value::Array(
        (0..layers)
            .map(|layer| {
                json!(if layer % 2 == 0 {
                    "sliding_attention"
                } else {
                    "full_attention"
                })
            })
            .collect(),
    );
    value
}

fn assert_contract(config: &DecoderConfig, layers: usize, experts: usize) -> Result<()> {
    assert_eq!(config.hidden_size, 2880);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.num_hidden_layers, layers);
    assert_eq!(config.num_experts, Some(experts));
    assert_eq!(config.top_k_experts, Some(4));
    assert!(config.attention_bias);
    assert!(!config.attention_sinks);
    assert!(config.layer_types.contains(&AttentionLayerType::Sliding));
    assert!(config.layer_types.contains(&AttentionLayerType::Full));
    let (factor, beta_fast, beta_slow, context, attention_factor) = config
        .rope_scaling
        .and_then(RopeScaling::yarn)
        .ok_or_else(|| ModelsError::InvalidConfig("expected YaRN scaling".into()))?;
    assert_eq!((factor, beta_fast, beta_slow, context), (32.0, 32.0, 1.0, 4096));
    assert!((attention_factor - 0.1_f64.mul_add(32.0_f64.ln(), 1.0)).abs() < 1.0e-12);
    Ok(())
}
