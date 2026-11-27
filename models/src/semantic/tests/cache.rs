//! What each layer caches per token, and how far back it attends.

use serde_json::json;

use super::{super::LayerCacheShape, catalog, *};

fn spec(config: &serde_json::Value) -> Result<SemanticModelSpec> {
    let decoder = DecoderConfig::from_value(config)?;
    SemanticModelSpec::discover(&decoder, &catalog(&["model.layers.0.input_layernorm.weight"]))
}

#[test]
fn layers_report_what_they_cache_per_token() -> Result<()> {
    let hybrid = spec(&json!({
        "model_type": "qwen3_next", "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "intermediate_size": 32, "vocab_size": 8, "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2, "linear_num_value_heads": 2, "linear_key_head_dim": 16,
        "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4
    }))?;
    assert_eq!(hybrid.decoder.layer(0)?.cache_shape(), None);
    assert_eq!(
        hybrid.decoder.layer(1)?.cache_shape(),
        Some(LayerCacheShape { heads: 2, key_width: 16, value_width: 16 })
    );
    // One shared `kv_lora_rank + rope_dim` row, no separate values.
    let latent = spec(&json!({
        "model_type": "deepseek_v2", "hidden_size": 64, "intermediate_size": 32,
        "moe_intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 4, "vocab_size": 8, "q_lora_rank": null, "kv_lora_rank": 32,
        "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
        "first_k_dense_replace": 1, "rms_norm_eps": 1e-6
    }))?;
    assert_eq!(
        latent.decoder.layer(0)?.cache_shape(),
        Some(LayerCacheShape { heads: 1, key_width: 40, value_width: 0 })
    );
    Ok(())
}

/// Llama 4's rotary layers attend within 32-token chunks; every fourth
/// layer, unrotated, sees the whole history.
#[test]
fn chunked_layers_attend_one_chunk_back() -> Result<()> {
    let spec = spec(&json!({
        "model_type": "llama4_text", "hidden_size": 64, "intermediate_size": 32,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "vocab_size": 32, "num_local_experts": 4, "num_experts_per_tok": 1,
        "attention_chunk_size": 32, "attn_temperature_tuning": true, "floor_scale": 32,
        "attn_scale": 0.1, "use_qk_norm": true
    }))?;
    assert_eq!(spec.decoder.layer(0)?.attention_span(), Some(32));
    assert_eq!(spec.decoder.layer(3)?.attention_span(), None);
    Ok(())
}
