//! How a layer encodes positions: rotary embeddings and their frequency
//! scaling.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "snake_case")]
pub enum PositionEncodingSpec {
    None,
    Rotary(RotarySpec),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotarySpec {
    pub theta: f64,
    pub partial_factor: f64,
    pub layout: RotaryLayoutSpec,
    /// Absent in sidecars written before it was typed, and for the default.
    #[serde(default, skip_serializing_if = "RotaryAlgorithm::is_default")]
    pub algorithm: RotaryAlgorithm,
    pub scaling: Option<RopeScalingSpec>,
}

/// The width a rotary layer spreads its pair frequencies over (the
/// configuration's per-layer `rope_type`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotaryAlgorithm {
    /// The rotated dimensions.
    #[default]
    Default,
    /// The whole head, of which only the leading pairs rotate (Gemma 4).
    Proportional,
}

impl RotaryAlgorithm {
    #[allow(clippy::trivially_copy_pass_by_ref)]
    const fn is_default(&self) -> bool {
        matches!(self, Self::Default)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "sections", rename_all = "snake_case")]
pub enum RotaryLayoutSpec {
    Standard,
    MultiSection(Vec<usize>),
    InterleavedMultiSection(Vec<usize>),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RopeScalingSpec {
    Linear {
        factor: f64,
    },
    PiecewiseFrequency {
        factor: f64,
        low_frequency_factor: f64,
        high_frequency_factor: f64,
        original_context_len: usize,
    },
    Yarn {
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        original_context_len: usize,
        attention_factor: f64,
        #[serde(default = "default_yarn_truncate")]
        truncate: bool,
    },
    /// Each pair's period multiplied by its factor (Phi-3 `LongRoPE`).
    LongRope {
        factors: crate::layout::RopeFactors,
        original_context_len: usize,
        attention_factor: f64,
    },
}

const fn default_yarn_truncate() -> bool {
    true
}
