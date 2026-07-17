use serde_json::Value;

use super::parse::{optional_usize, require_usize};
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionOutput {
    Direct,
    Gated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearAttentionConfig {
    pub convolution_kernel_size: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RotaryEmbeddingLayout {
    Standard,
    MultiSection(Vec<usize>),
    InterleavedMultiSection(Vec<usize>),
}

pub(super) fn linear_attention(value: &Value) -> Result<Option<LinearAttentionConfig>> {
    let Some(convolution_kernel_size) = optional_usize(value, &["linear_conv_kernel_dim"])? else {
        return Ok(None);
    };
    Ok(Some(LinearAttentionConfig {
        convolution_kernel_size,
        key_heads: require_usize(value, &["linear_num_key_heads"])?,
        value_heads: require_usize(value, &["linear_num_value_heads"])?,
        key_head_dim: require_usize(value, &["linear_key_head_dim"])?,
        value_head_dim: require_usize(value, &["linear_value_head_dim"])?,
    }))
}

pub(super) fn attention_output(value: &Value) -> Result<AttentionOutput> {
    Ok(if optional_bool(value, "attn_output_gate")?.unwrap_or(false) {
        AttentionOutput::Gated
    } else {
        AttentionOutput::Direct
    })
}

pub(super) fn rope_layout(value: &Value) -> Result<RotaryEmbeddingLayout> {
    let sections = rope_sections(value)?;
    let interleaved = rope_interleaved(value)?;
    match (sections, interleaved) {
        (None, false) => Ok(RotaryEmbeddingLayout::Standard),
        (Some(sections), false) => Ok(RotaryEmbeddingLayout::MultiSection(sections)),
        (Some(sections), true) => Ok(RotaryEmbeddingLayout::InterleavedMultiSection(sections)),
        (None, true) => Err(super::parse::invalid("mrope_interleaved requires mrope_section")),
    }
}

fn rope_sections(value: &Value) -> Result<Option<Vec<usize>>> {
    let Some(raw) = value
        .get("rope_parameters")
        .and_then(|parameters| parameters.get("mrope_section"))
    else {
        return Ok(None);
    };
    let Some(items) = raw.as_array() else {
        return Err(super::parse::invalid("mrope_section must be an array"));
    };
    let sections = items
        .iter()
        .map(|item| {
            let number = item.as_u64().ok_or_else(|| {
                super::parse::invalid("mrope_section entries must be unsigned integers")
            })?;
            Ok(usize::try_from(number)?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(sections))
}

fn rope_interleaved(value: &Value) -> Result<bool> {
    let Some(raw) = value
        .get("rope_parameters")
        .and_then(|parameters| parameters.get("mrope_interleaved"))
    else {
        return Ok(false);
    };
    raw.as_bool()
        .ok_or_else(|| super::parse::invalid("mrope_interleaved must be a boolean"))
}

fn optional_bool(value: &Value, key: &str) -> Result<Option<bool>> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    raw.as_bool()
        .map(Some)
        .ok_or_else(|| super::parse::invalid(format!("{key} must be a boolean")))
}
