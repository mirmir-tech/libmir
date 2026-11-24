//! Metal keeps one chunk of K/V for Llama 4's chunked layers.

use serde_json::json;

use super::{decoder_budget, *};

#[test]
fn metal_bounds_chunked_layers_to_their_chunk() -> crate::Result<()> {
    // Llama 4: layers 0-2 attend in chunks of 32 tokens, layer 3 globally.
    let decoder = DecoderConfig::from_value(&json!({
        "model_type": "llama4_text", "hidden_size": 64, "intermediate_size": 32,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "vocab_size": 32, "num_local_experts": 4, "num_experts_per_tok": 1,
        "attention_chunk_size": 32, "attn_temperature_tuning": true, "floor_scale": 32,
        "attn_scale": 0.1, "use_qk_norm": true
    }))?;
    let config = RuntimeConfig::default();
    let (metal, _) = decoder_budget(&decoder, &config, &BackendTarget::Metal)?;
    let (cuda, _) = decoder_budget(&decoder, &config, &BackendTarget::Cuda)?;
    // Only the global layer grows with the sequence on Metal.
    assert_eq!(cuda, 4 * metal);
    Ok(())
}
