//! Granite's scaling multipliers (`granite`, `granitemoe`): llama.cpp's
//! `embedding_scale`, `attention.scale`, `residual_scale` and `logit_scale`
//! under their Hugging Face names. Granite divides logits by `logit_scale`
//! (`logits_scaling`); other architectures' `logit_scale` multiplies, so the
//! mapping is Granite's own.

use serde_json::{Map, Value, json};

use super::{super::GgufHeader, decimal};

pub(super) fn granite(header: &GgufHeader, architecture: &str, config: &mut Map<String, Value>) {
    if !matches!(architecture, "granite" | "granitemoe") {
        return;
    }
    for (name, field) in [
        ("embedding_scale", "embedding_multiplier"),
        ("attention.scale", "attention_multiplier"),
        ("residual_scale", "residual_multiplier"),
        ("logit_scale", "logits_scaling"),
    ] {
        // llama.cpp reads 0 as "not set".
        if let Some(value) = header
            .get(&format!("{architecture}.{name}"))
            .and_then(decimal)
            .filter(|value| *value != 0.0)
        {
            config.insert(field.into(), json!(value));
        }
    }
}
