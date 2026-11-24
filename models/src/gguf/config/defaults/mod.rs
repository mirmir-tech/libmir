//! Import defaults for what GGUF leaves undeclared, keyed by the declared
//! `general.architecture` exactly as `layout/decoder/defaults.rs` keys
//! `SafeTensors` defaults by `model_type`. Each value names its source; an
//! architecture missing here is refused at load instead of guessed.

use serde_json::{Map, Value, json};

use crate::layout::{QUERY_KEY_ROWS, QUERY_KEY_ROWS_INTERLEAVED};

/// The Hugging Face `model_type` of a GGUF architecture, which keys the
/// decoder defaults and model-specific generation (gpt-oss Harmony). GGUF
/// spells most architectures as HF does; the exceptions are listed.
pub(super) fn model_type(architecture: &str) -> &str {
    match architecture {
        "gpt-oss" => "gpt_oss",
        "qwen3moe" => "qwen3_moe",
        "qwen2moe" => "qwen2_moe",
        "qwen35" => "qwen3_5_text",
        "gemma4" => "gemma4_text",
        "command-r" => "cohere",
        "glm4moe" => "glm4_moe",
        "llama4" => "llama4_text",
        "qwen35moe" => "qwen3_5_moe_text",
        other => other,
    }
}

/// The MLP activation, which llama.cpp fixes in each architecture's graph
/// (`src/models/*.cpp`, v1.0.0) instead of declaring it.
pub(super) fn activation(architecture: &str) -> Option<&'static str> {
    match architecture {
        "llama" | "mistral" | "qwen2" | "qwen3" | "qwen3moe" | "gpt-oss" | "deepseek2"
        | "mistral4" | "qwen35" | "qwen35moe" | "phi3" | "granite" | "granitemoe" | "glm4"
        | "glm4moe" | "phimoe" | "llama4" | "olmo2" | "command-r" | "cohere2" => Some("silu"),
        "gemma2" | "gemma3" | "gemma4" => Some("gelu_pytorch_tanh"),
        _ => None,
    }
}

/// Whether the import derives a configuration for `architecture`: those it
/// knows the MLP activation of, and Nemotron-H, whose mixers declare theirs.
pub(in crate::gguf) fn imports(architecture: &str) -> bool {
    activation(architecture).is_some() || architecture == "nemotron_h"
}

/// Fields the architecture fixes in code rather than metadata.
pub(super) fn fixed(architecture: &str, config: &mut Map<String, Value>) {
    if query_key_rows_interleaved(architecture) {
        config.insert(QUERY_KEY_ROWS.into(), json!(QUERY_KEY_ROWS_INTERLEAVED));
    }
    if architecture == "gemma2" {
        // llama.cpp `src/models/gemma2-iswa.cpp`: scores scale by `1 /
        // sqrt(head_dim)`, except the 27B model (46 layers), which scales by
        // `1 / sqrt(n_embd / n_head)`; embeddings scale by `sqrt(n_embd)`; the
        // converter already added one to every norm weight.
        let unsigned = |key: &str| config.get(key).and_then(Value::as_u64);
        let scalar = match (
            unsigned("num_hidden_layers"),
            unsigned("hidden_size"),
            unsigned("num_attention_heads"),
        ) {
            (Some(46), Some(hidden), Some(heads)) if heads > 0 => Some(hidden / heads),
            _ => unsigned("head_dim"),
        };
        if let Some(scalar) = scalar {
            config.insert("query_pre_attn_scalar".into(), json!(scalar));
        }
        if let Some(hidden) = config.get("hidden_size").and_then(Value::as_f64) {
            config.insert("embedding_multiplier".into(), json!(hidden.sqrt()));
        }
    }
    if architecture == "gemma3" {
        // llama.cpp v1.0.0 `src/models/gemma3.cpp`: sliding layers rotate with
        // base 10000; scores scale by `1 / sqrt(head_dim)`, except the 27B
        // model (62 layers, which GGUF does not otherwise record), which
        // scales by `1 / sqrt(n_embd / n_head)` as google/gemma-3-27b-it's
        // `query_pre_attn_scalar` of 168 does; embeddings scale by
        // `sqrt(n_embd)`. The converter already added one to every norm
        // weight, so the norms are plain.
        config.insert("rope_local_base_freq".into(), json!(10_000.0));
        let unsigned = |key: &str| config.get(key).and_then(Value::as_u64);
        let scalar = match (
            unsigned("num_hidden_layers"),
            unsigned("hidden_size"),
            unsigned("num_attention_heads"),
        ) {
            (Some(62), Some(hidden), Some(heads)) if heads > 0 => Some(hidden / heads),
            _ => unsigned("head_dim"),
        };
        if let Some(scalar) = scalar {
            config.insert("query_pre_attn_scalar".into(), json!(scalar));
        }
        if let Some(hidden) = config.get("hidden_size").and_then(Value::as_f64) {
            config.insert("embedding_multiplier".into(), json!(hidden.sqrt()));
        }
    }
    if architecture == "gpt-oss" {
        // llama.cpp `LLM_FFN_SWIGLU_OAI_MOE`: alpha 1.702, limit 7.0, the
        // values of openai/gpt-oss-20b `config.json` `swiglu_limit`.
        config.insert("swiglu_limit".into(), json!(7.0));
    }
}

/// llama.cpp's converter permutes the query and key rows of these
/// architectures so its pairwise `RoPE` sees Hugging Face's `rotate_half`
/// halves (`convert_hf_to_gguf.py` `LlamaModel.permute`, v1.0.0); Mistral
/// models convert as `llama`.
fn query_key_rows_interleaved(architecture: &str) -> bool {
    // GLM-4, Cohere and Llama 4 rotate adjacent pairs themselves (llama.cpp
    // `LLAMA_ROPE_TYPE_NORM` on unpermuted rows), which is the same rotation.
    matches!(
        architecture,
        "llama" | "granite" | "granitemoe" | "glm4" | "command-r" | "cohere2" | "llama4"
    )
}

/// `YaRN` correction-dimension truncation, which GGUF does not record.
/// openai/gpt-oss-* `config.json` sets `truncate: false`; llama.cpp v1.0.0
/// always truncates, so its perplexity on these files differs slightly.
pub(super) fn yarn_truncate(architecture: &str) -> Option<bool> {
    (architecture == "gpt-oss").then_some(false)
}

#[cfg(test)]
mod tests;
