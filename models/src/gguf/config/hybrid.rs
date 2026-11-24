//! Qwen3.5/3.6/3.8 hybrid decoders (`qwen35`, `qwen35moe`): Gated `DeltaNet`
//! layers with a full-attention layer every `full_attention_interval`, as the
//! Hugging Face `text_config` declares them.
//!
//! llama.cpp's converter (`conversion/qwen.py`, `Qwen3NextModel` and
//! `_LinearAttentionVReorderBase`) changes the stored tensors: `A_log` becomes
//! `ssm_a = -exp(A_log)`, `conv1d` drops its unit axis, every norm except the
//! gated `DeltaNet` output norm carries its `1 +` offset, and value heads move
//! from grouped to tiled order when there are more value than key heads. The
//! [`LINEAR_ATTENTION_LAYOUT`] key tells the runtime to undo that.

use serde_json::{Map, Value, json};

use super::super::GgufHeader;
use crate::{
    error::{ModelsError, Result},
    layout::{LINEAR_ATTENTION_LAYOUT, LINEAR_ATTENTION_LAYOUT_LLAMA_CPP},
};

/// Whether `architecture` is a hybrid `DeltaNet` decoder; fills its keys.
pub(super) fn hybrid(
    header: &GgufHeader,
    architecture: &str,
    config: &mut Map<String, Value>,
) -> Result<bool> {
    if !matches!(architecture, "qwen35" | "qwen35moe") {
        return Ok(false);
    }
    let key = |name: &str| format!("{architecture}.{name}");
    let unsigned = |name: &str| {
        header
            .unsigned(&key(name))
            .ok_or_else(|| ModelsError::InvalidGguf(format!("{architecture}: {name} is missing")))
    };
    let value_heads = unsigned("ssm.time_step_rank")?;
    let inner = unsigned("ssm.inner_size")?;
    if value_heads == 0 || !inner.is_multiple_of(value_heads) {
        return Err(ModelsError::InvalidGguf(format!(
            "{architecture}: ssm.inner_size {inner} does not split into {value_heads} heads"
        )));
    }
    config.insert("linear_conv_kernel_dim".into(), json!(unsigned("ssm.conv_kernel")?));
    config.insert("linear_num_key_heads".into(), json!(unsigned("ssm.group_count")?));
    config.insert("linear_num_value_heads".into(), json!(value_heads));
    config.insert("linear_key_head_dim".into(), json!(unsigned("ssm.state_size")?));
    config.insert("linear_value_head_dim".into(), json!(inner / value_heads));
    config.insert(LINEAR_ATTENTION_LAYOUT.into(), json!(LINEAR_ATTENTION_LAYOUT_LLAMA_CPP));
    // Full-attention layers gate their output with the second half of q_proj.
    config.insert("attn_output_gate".into(), json!(true));
    let layers = config.get("num_hidden_layers").and_then(Value::as_u64).ok_or_else(|| {
        ModelsError::InvalidGguf(format!("{architecture}: block_count is missing"))
    })?;
    let interval = unsigned("full_attention_interval")?.max(1);
    let layer_types = (1..=layers)
        .map(|layer| {
            if layer.is_multiple_of(interval) {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect::<Vec<_>>();
    config.insert("layer_types".into(), json!(layer_types));
    // Rotary embedding covers `rope.dimension_count` of each head, in
    // interleaved multi-section form (text positions repeat in every section).
    let head_dim = unsigned("attention.key_length")?;
    let rotary = unsigned("rope.dimension_count")?;
    if head_dim == 0 || rotary == 0 || rotary > head_dim {
        return Err(ModelsError::InvalidGguf(format!(
            "{architecture}: rope.dimension_count {rotary} does not fit head_dim {head_dim}"
        )));
    }
    let factor = f64::from(u32::try_from(rotary)?) / f64::from(u32::try_from(head_dim)?);
    config.insert("partial_rotary_factor".into(), json!(factor));
    let mut parameters = Map::new();
    parameters.insert("rope_type".into(), json!("default"));
    parameters.insert("partial_rotary_factor".into(), json!(factor));
    if let Some(theta) = config.get("rope_theta").cloned() {
        parameters.insert("rope_theta".into(), theta);
    }
    let sections = header
        .get(&key("rope.dimension_sections"))
        .and_then(super::super::MetadataValue::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(super::super::MetadataValue::as_u64)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let sections = sections.into_iter().filter(|section| *section > 0).collect::<Vec<_>>();
    if !sections.is_empty() {
        parameters.insert("mrope_section".into(), json!(sections));
        parameters.insert("mrope_interleaved".into(), json!(true));
    }
    config.insert("rope_parameters".into(), Value::Object(parameters));
    Ok(true)
}
