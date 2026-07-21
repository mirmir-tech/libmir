use serde_json::{Value, json};

use super::super::{AttentionLayerType, DecoderConfig, RopeScaling};
use crate::error::Result;

#[test]
fn reads_official_20b_geometry() -> Result<()> {
    let config = DecoderConfig::from_value(&configuration(24, 32))?;

    assert_contract(&config, 24, 32);
    Ok(())
}

#[test]
fn reads_official_120b_geometry() -> Result<()> {
    let config = DecoderConfig::from_value(&configuration(36, 128))?;

    assert_contract(&config, 36, 128);
    Ok(())
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

fn assert_contract(config: &DecoderConfig, layers: usize, experts: usize) {
    assert_eq!(config.hidden_size, 2880);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.num_hidden_layers, layers);
    assert_eq!(config.num_experts, Some(experts));
    assert_eq!(config.top_k_experts, Some(4));
    assert!(config.attention_bias);
    assert!(!config.attention_sinks);
    assert!(config.layer_types.contains(&AttentionLayerType::Sliding));
    assert!(config.layer_types.contains(&AttentionLayerType::Full));
    assert_eq!(config.rope_scaling.and_then(RopeScaling::yarn), Some((32.0, 32.0, 1.0, 4096)));
}
