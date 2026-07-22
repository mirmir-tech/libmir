use serde_json::json;

use super::*;
use crate::layout::{ModelLayout, ModelMetadata};

#[test]
fn reads_official_mistral_7b_v03_geometry() -> Result<()> {
    let config = DecoderConfig::from_value(&json!({
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32768,
        "max_position_embeddings": 32768,
        "sliding_window": null,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 0.00001,
        "hidden_act": "silu",
        "tie_word_embeddings": false
    }))?;

    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.max_position_embeddings, 32768);
    assert_eq!(config.sliding_window, None);
    assert_eq!(config.rope_theta, Some(1_000_000.0));
    assert!(!config.tie_word_embeddings);
    Ok(())
}

#[test]
fn reads_official_ministral_8b_interleaved_attention() -> Result<()> {
    let layer_types = (0..36)
        .map(|layer| {
            if layer % 4 == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect::<Vec<_>>();
    let config = DecoderConfig::from_value(&json!({
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 131_072,
        "max_position_embeddings": 32768,
        "sliding_window": 32768,
        "rope_theta": 100_000_000.0,
        "rms_norm_eps": 0.00001,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
        "layer_types": layer_types
    }))?;

    assert_eq!(config.num_hidden_layers, 36);
    assert_eq!(config.intermediate_size, 12288);
    assert_eq!(config.vocab_size, 131_072);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.rope_theta, Some(100_000_000.0));
    assert_eq!(config.layer_type(0), AttentionLayerType::Full);
    assert_eq!(config.layer_type(1), AttentionLayerType::Sliding);
    assert_eq!(config.layer_type(35), AttentionLayerType::Sliding);
    assert_eq!(config.layer_sliding_window(0), None);
    assert_eq!(config.layer_sliding_window(1), Some(32768));
    Ok(())
}

#[test]
fn extends_ministral_context_from_official_params() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-ministral-params-{}", std::process::id()));
    std::fs::create_dir(&root)?;
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&json!({
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 64,
            "max_position_embeddings": 32768
        }))?,
    )?;
    std::fs::write(
        root.join("params.json"),
        serde_json::to_vec(&json!({"max_position_embeddings": 131_072}))?,
    )?;
    std::fs::write(root.join("model.safetensors"), [])?;

    let layout = ModelLayout::inspect(&root)?;
    let config = DecoderConfig::from_layout(&layout)?;
    let metadata = ModelMetadata::from_layout(&layout)?;
    std::fs::remove_dir_all(root)?;

    assert_eq!(config.max_position_embeddings, 131_072);
    assert_eq!(metadata.context_len, 131_072);
    Ok(())
}
