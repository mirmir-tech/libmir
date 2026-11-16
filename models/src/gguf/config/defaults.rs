//! Import defaults for what GGUF leaves undeclared, keyed by the declared
//! `general.architecture` exactly as `layout/decoder/defaults.rs` keys
//! `SafeTensors` defaults by `model_type`. Each value names its source; an
//! architecture missing here is refused at load instead of guessed.

use serde_json::{Map, Value, json};

/// The Hugging Face `model_type` of a GGUF architecture, which keys the
/// decoder defaults and model-specific generation (gpt-oss Harmony). GGUF
/// spells most architectures as HF does; the exceptions are listed.
pub(super) fn model_type(architecture: &str) -> &str {
    match architecture {
        "gpt-oss" => "gpt_oss",
        "qwen3moe" => "qwen3_moe",
        "qwen2moe" => "qwen2_moe",
        other => other,
    }
}

/// The MLP activation, which llama.cpp fixes in each architecture's graph
/// (`src/models/*.cpp`, v1.0.0) instead of declaring it.
pub(super) fn activation(architecture: &str) -> Option<&'static str> {
    match architecture {
        "llama" | "mistral" | "qwen2" | "qwen3" | "qwen3moe" | "gpt-oss" => Some("silu"),
        _ => None,
    }
}

/// Fields the architecture fixes in code rather than metadata.
pub(super) fn fixed(architecture: &str, config: &mut Map<String, Value>) {
    if architecture == "gpt-oss" {
        // llama.cpp `LLM_FFN_SWIGLU_OAI_MOE`: alpha 1.702, limit 7.0, the
        // values of openai/gpt-oss-20b `config.json` `swiglu_limit`.
        config.insert("swiglu_limit".into(), json!(7.0));
    }
}

/// `YaRN` correction-dimension truncation, which GGUF does not record.
/// openai/gpt-oss-* `config.json` sets `truncate: false`; llama.cpp v1.0.0
/// always truncates, so its perplexity on these files differs slightly.
pub(super) fn yarn_truncate(architecture: &str) -> Option<bool> {
    (architecture == "gpt-oss").then_some(false)
}
