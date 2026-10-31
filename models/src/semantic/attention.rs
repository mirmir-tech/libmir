use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "snake_case")]
pub enum MixerSpec {
    SoftmaxAttention(AttentionSpec),
    LinearAttention(LinearAttentionSpec),
    StateSpace(StateSpaceSpec),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionSpec {
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub key_value_relation: KeyValueRelation,
    pub qk_normalization: QkNormalization,
    pub projection_bias: bool,
    pub output: AttentionOutputSpec,
    pub sinks: bool,
    pub scale: f64,
    pub window: Option<usize>,
    pub position: PositionEncodingSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearAttentionSpec {
    pub convolution_kernel_size: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub output: AttentionOutputSpec,
}

/// Mamba-2 selective state-space mixer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateSpaceSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub state_size: usize,
    pub groups: usize,
    pub convolution_kernel_size: usize,
    pub convolution_bias: bool,
    pub projection_bias: bool,
    pub chunk_size: usize,
    pub time_step_minimum: f64,
    /// Upper clamp of the time step; `None` leaves it unbounded.
    pub time_step_maximum: Option<f64>,
    pub state_dtype: StateSpaceStateDTypeSpec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateSpaceStateDTypeSpec {
    Float32,
    BFloat16,
    Float16,
}

impl StateSpaceSpec {
    /// Width of the gated SSM output.
    #[must_use]
    pub const fn inner_size(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Channels passing through the causal convolution.
    #[must_use]
    pub const fn convolution_channels(&self) -> usize {
        self.inner_size() + 2 * self.groups * self.state_size
    }

    /// Width of the fused `z | xBC | dt` input projection.
    #[must_use]
    pub const fn input_projection_size(&self) -> usize {
        self.inner_size() + self.convolution_channels() + self.heads
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyValueRelation {
    Separate,
    KeyEqualsValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QkNormalization {
    None,
    QueryKeyRms,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionOutputSpec {
    Direct,
    Gated,
}

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
    pub algorithm: Option<String>,
    pub scaling: Option<RopeScalingSpec>,
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
}

const fn default_yarn_truncate() -> bool {
    true
}
