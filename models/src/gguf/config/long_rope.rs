//! Phi-3 `LongRoPE`, which llama.cpp's converter stores as
//! `rope_factors_long` and `rope_factors_short` tensors beside
//! `rope.scaling.original_context_length` and `rope.scaling.attn_factor`
//! (`convert_hf_to_gguf.py` `Phi3MiniModel.generate_extra_tensors`, v1.0.0).

use serde_json::{Map, Value, json};

use super::super::{ConfigurationVector, GgufModel};
use crate::error::{ModelsError, Result};

const LONG: &str = ConfigurationVector::LongRopeFactors.tensor();
const SHORT: &str = ConfigurationVector::ShortRopeFactors.tensor();
/// microsoft/Phi-3.5-MoE-instruct `rope_scaling.long_mscale`.
const PHIMOE_MSCALE: f64 = 1.243_163_121_016_122;

/// Writes the `longrope` `rope_scaling` the source `config.json` declared.
pub(super) fn rope_scaling(
    model: &GgufModel,
    architecture: &str,
    config: &mut Map<String, Value>,
) -> Result<()> {
    let (long, short) = match (
        model.vector(ConfigurationVector::LongRopeFactors),
        model.vector(ConfigurationVector::ShortRopeFactors),
    ) {
        (None, None) => return Ok(()),
        (Some(long), Some(short)) => (long?, short?),
        _ => {
            return Err(ModelsError::InvalidGguf(format!(
                "{architecture}: LongRoPE needs both {LONG} and {SHORT}"
            )));
        },
    };
    let header = model.header();
    let key = |name: &str| format!("{architecture}.rope.scaling.{name}");
    let original = header.unsigned(&key("original_context_length")).ok_or_else(|| {
        ModelsError::InvalidGguf(format!(
            "{architecture}: LongRoPE needs rope.scaling.original_context_length"
        ))
    })?;
    let mut scaling = json!({
        "rope_type": "longrope", "long_factor": long, "short_factor": short,
        "original_max_position_embeddings": original,
    });
    // The converter writes `sqrt(1 + ln(s) / ln(original))` as `attn_factor`,
    // but Phi-3.5-MoE declares its own scale, which transformers and MLX-LM
    // apply (microsoft/Phi-3.5-MoE-instruct `config.json`).
    if architecture == "phimoe" {
        scaling["long_mscale"] = json!(PHIMOE_MSCALE);
        scaling["short_mscale"] = json!(PHIMOE_MSCALE);
    } else if let Some(factor) = header.get(&key("attn_factor")).and_then(super::decimal) {
        scaling["attention_factor"] = json!(factor);
    }
    config.insert("original_max_position_embeddings".into(), json!(original));
    config.insert("rope_scaling".into(), scaling);
    Ok(())
}
