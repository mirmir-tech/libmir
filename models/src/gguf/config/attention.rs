//! `RoPE` scaling (`YaRN`, linear) and sliding-window layers from
//! `<arch>.rope.scaling.*` and
//! `<arch>.attention.sliding_window[_pattern]`.

use serde_json::{Map, Value, json};

use super::{super::GgufHeader, decimal, defaults};

/// `rope_scaling` for a `YaRN` model: llama.cpp's `rope.scaling.*` keys under
/// their Hugging Face names.
pub(super) fn rope_scaling(
    header: &GgufHeader,
    architecture: &str,
    config: &mut Map<String, Value>,
) {
    let key = |name: &str| format!("{architecture}.rope.scaling.{name}");
    if header.string(&key("type")) == Some("linear") {
        if let Some(factor) = header.get(&key("factor")).and_then(decimal) {
            config.insert("rope_scaling".into(), json!({"rope_type": "linear", "factor": factor}));
        }
        return;
    }
    if header.string(&key("type")) != Some("yarn") {
        return;
    }
    let mut scaling = Map::new();
    scaling.insert("rope_type".into(), json!("yarn"));
    for (name, field) in [
        ("factor", "factor"),
        ("yarn_beta_fast", "beta_fast"),
        ("yarn_beta_slow", "beta_slow"),
        ("yarn_attn_factor", "attention_factor"),
    ] {
        if let Some(value) = header.get(&key(name)).and_then(decimal) {
            scaling.insert(field.into(), json!(value));
        }
    }
    if let Some(length) = header.unsigned(&key("original_context_length")) {
        scaling.insert("original_max_position_embeddings".into(), json!(length));
    }
    if let Some(truncate) = defaults::yarn_truncate(architecture) {
        scaling.insert("truncate".into(), json!(truncate));
    }
    config.insert("rope_scaling".into(), Value::Object(scaling));
}

/// `layer_types` for a model with a sliding window: llama.cpp's
/// `set_swa_pattern(n)` makes layer `i` sliding when `i % n < n - 1`, with
/// `n` from `attention.sliding_window_pattern` (default 2 for gpt-oss, 6 for
/// Gemma 3: `llama-model.cpp` `LLM_ARCH_GEMMA3`, v1.0.0).
pub(super) fn layer_types(
    header: &GgufHeader,
    architecture: &str,
    config: &mut Map<String, Value>,
) {
    let key = |name: &str| format!("{architecture}.{name}");
    let (Some(_), Some(layers)) = (
        header.unsigned(&key("attention.sliding_window")),
        header.unsigned(&key("block_count")),
    ) else {
        return;
    };
    let pattern =
        header
            .unsigned(&key("attention.sliding_window_pattern"))
            .or(match architecture {
                "gpt-oss" => Some(2),
                "gemma3" => Some(6),
                _ => None,
            });
    let Some(pattern) = pattern.filter(|&pattern| pattern > 0) else {
        return;
    };
    let types: Vec<&str> = (0..layers)
        .map(|layer| {
            if layer % pattern < pattern - 1 {
                "sliding_attention"
            } else {
                "full_attention"
            }
        })
        .collect();
    config.insert("layer_types".into(), json!(types));
}
