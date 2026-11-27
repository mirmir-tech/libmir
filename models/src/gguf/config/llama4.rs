//! Llama 4 (`llama4`) under transformers `Llama4TextConfig` names.
//!
//! llama.cpp writes the dense MLP width as `feed_forward_length` and the
//! routed and shared experts' as `expert_feed_forward_length`, and the
//! chunked-attention size as `attention.sliding_window`
//! (`llama_model_llama4::load_arch_hparams`, v1.0.0).

use serde_json::{Map, Value, json};

use super::super::GgufHeader;

pub(super) fn llama4(header: &GgufHeader, architecture: &str, config: &mut Map<String, Value>) {
    if architecture != "llama4" {
        return;
    }
    let key = |name: &str| format!("{architecture}.{name}");
    if let Some(dense) = config.remove("intermediate_size") {
        config.insert("intermediate_size_mlp".into(), dense);
    }
    if let Some(experts) = config.remove("moe_intermediate_size") {
        config.insert("intermediate_size".into(), experts);
    }
    // An explicit zero window means no chunking (llama.cpp's
    // `LLAMA_SWA_TYPE_NONE`); otherwise llama.cpp chunks by 8192 tokens,
    // whatever other window the file declares, and tunes the temperature
    // with that size, as Llama 4 Scout's `config.json` does. Importing the
    // declared window would run such a file unlike llama.cpp does.
    drop(config.remove("sliding_window"));
    let chunk = match header.unsigned(&key("attention.sliding_window")) {
        Some(0) => None,
        Some(_) | None => Some(8192),
    };
    config.insert("attention_chunk_size".into(), json!(chunk.unwrap_or(0)));
    if let Some(step) = header.unsigned(&key("interleave_moe_layer_step")) {
        config.insert("interleave_moe_layer_step".into(), json!(step));
    }
    // llama.cpp normalizes Q/K except for the 128-expert Maverick, whose
    // `config.json` sets `use_qk_norm: false`.
    let experts = header.unsigned(&key("expert_count"));
    config.insert("use_qk_norm".into(), json!(experts != Some(128)));
    config.insert("attn_temperature_tuning".into(), json!(chunk.is_some()));
}
