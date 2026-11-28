//! Both Metal and CUDA keep Llama 4's chunked layers in paged K/V storage.

use models::layout::DecoderConfig;
use serde_json::json;

use super::{decoder_budget, *};

#[test]
fn chunked_layers_use_paged_storage_on_both_metal_and_cuda() -> crate::Result<()> {
    // Llama 4: layers 0-2 attend in chunks of 32 tokens, layer 3 globally.
    let decoder = DecoderConfig::from_value(&json!({
        "model_type": "llama4_text", "hidden_size": 64, "intermediate_size": 32,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "vocab_size": 32, "num_local_experts": 4, "num_experts_per_tok": 1,
        "attention_chunk_size": 32, "attn_temperature_tuning": true, "floor_scale": 32,
        "attn_scale": 0.1, "use_qk_norm": true
    }))?;
    let config = RuntimeConfig::default();
    let (metal_per_token, metal_total) = decoder_budget(&decoder, &config, BackendTarget::Metal)?;
    let (cuda_per_token, cuda_total) = decoder_budget(&decoder, &config, BackendTarget::Cuda)?;
    // Every layer (chunked and global) grows with paged K/V on both backends.
    assert!(metal_per_token > 0);
    assert_eq!(metal_per_token, cuda_per_token);
    assert_eq!(metal_total, cuda_total);
    Ok(())
}
