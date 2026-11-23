use serde_json::Value;

use super::parse::{invalid, optional_string, optional_usize, require_usize};
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
    pub layout: LinearAttentionLayout,
}

/// How the stored `DeltaNet` tensors relate to the Hugging Face modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearAttentionLayout {
    /// A Hugging Face checkpoint or its MLX conversion.
    Checkpoint,
    /// llama.cpp's GGUF conversion: `ssm_a = -exp(A_log)`, a `[channels,
    /// kernel]` convolution, unit offsets folded into every norm except the
    /// gated output norm, and value heads tiled across key heads
    /// (`[K0, K1, .., K0, K1, ..]`) when there are more value than key heads.
    LlamaCpp,
}

/// The `config.json` key GGUF imports write for llama.cpp's layout.
pub const LINEAR_ATTENTION_LAYOUT: &str = "mir_linear_attention_layout";
pub const LINEAR_ATTENTION_LAYOUT_LLAMA_CPP: &str = "llama_cpp";

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
        layout: match optional_string(value, LINEAR_ATTENTION_LAYOUT)?.as_deref() {
            None => LinearAttentionLayout::Checkpoint,
            Some(LINEAR_ATTENTION_LAYOUT_LLAMA_CPP) => LinearAttentionLayout::LlamaCpp,
            Some(other) => {
                return Err(invalid(format!("unknown {LINEAR_ATTENTION_LAYOUT} `{other}`")));
            },
        },
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
        (None, true) => Err(invalid("mrope_interleaved requires mrope_section")),
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
        return Err(invalid("mrope_section must be an array"));
    };
    let sections = items
        .iter()
        .map(|item| {
            let number = item
                .as_u64()
                .ok_or_else(|| invalid("mrope_section entries must be unsigned integers"))?;
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
    raw.as_bool().ok_or_else(|| invalid("mrope_interleaved must be a boolean"))
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
        .ok_or_else(|| invalid(format!("{key} must be a boolean")))
}

/// How routed experts are selected and weighed.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ExpertRouter {
    /// Softmax top-k renormalized over the selection.
    #[default]
    SoftmaxTopK,
    /// Phi-3.5-MoE's sparsemixer with `router_jitter_noise`.
    SparseMixer { jitter: f64 },
    /// Top logits whose sigmoid scales the expert input (Llama 4).
    SigmoidInputScaled,
}
