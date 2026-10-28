mod raw;

use std::{fs, path::Path};

use crate::error::{ModelsError, Result};

/// Attention reach of one `ModernBERT` layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModernBertAttention {
    /// Every token attends to every valid token.
    Global,
    /// Token `i` attends to keys `j` with `|i - j| <= radius`.
    Local { radius: usize },
}

/// A bidirectional `ModernBERT` encoder: pre-norm layers without biases,
/// alternating global and banded local attention with per-kind rotary bases,
/// and a `GeGLU` feed-forward block.
#[derive(Debug, Clone, PartialEq)]
pub struct ModernBertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    /// Attention reach of every layer, in order.
    pub layers: Vec<ModernBertAttention>,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
}

impl ModernBertConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json(&fs::read_to_string(path)?)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str::<raw::RawConfig>(json)?.try_into()
    }

    #[must_use]
    pub const fn rope_theta(&self, attention: ModernBertAttention) -> f64 {
        match attention {
            ModernBertAttention::Global => self.global_rope_theta,
            ModernBertAttention::Local { .. } => self.local_rope_theta,
        }
    }
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests;
