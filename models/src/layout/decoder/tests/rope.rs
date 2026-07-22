use serde_json::json;

use super::{DecoderConfig, RopeScaling};
use crate::error::{ModelsError, Result};

#[test]
fn reads_deepseek_qwen_yarn_defaults_and_legacy_attention_multiplier() -> Result<()> {
    let config = DecoderConfig::from_value(&decoder(&json!({
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "attn_factor": 0.878_248_856_286_941_9
    })))?;
    let Some(RopeScaling::Yarn {
        factor,
        beta_fast,
        beta_slow,
        original_context_len,
        attention_factor,
    }) = config.rope_scaling
    else {
        return Err(ModelsError::InvalidConfig("expected YaRN scaling".into()));
    };

    assert!(close(factor, 4.0));
    assert!(close(beta_fast, 32.0));
    assert!(close(beta_slow, 1.0));
    assert_eq!(original_context_len, 32768);
    assert!((attention_factor - 1.0).abs() < 1.0e-12);
    Ok(())
}

#[test]
fn prefers_explicit_yarn_attention_factor() -> Result<()> {
    let config = DecoderConfig::from_value(&decoder(&json!({
        "rope_type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "attention_factor": 1.25,
        "attn_factor": 0.5
    })))?;

    assert_eq!(
        config.rope_scaling.and_then(RopeScaling::yarn),
        Some((8.0, 32.0, 1.0, 8192, 1.25))
    );
    Ok(())
}

#[test]
fn derives_yarn_attention_factor_from_mscale_pair() -> Result<()> {
    let config = DecoderConfig::from_value(&decoder(&json!({
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "mscale": 1.0,
        "mscale_all_dim": 1.0
    })))?;
    let attention_factor = config.rope_scaling.and_then(RopeScaling::yarn).map(|values| values.4);

    assert_eq!(attention_factor, Some(1.0));
    Ok(())
}

fn decoder(rope_scaling: &serde_json::Value) -> serde_json::Value {
    json!({
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 32,
        "rope_scaling": rope_scaling
    })
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1.0e-12
}
