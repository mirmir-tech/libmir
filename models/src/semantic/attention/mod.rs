mod position;
mod window;

pub use position::{
    PositionEncodingSpec, RopeScalingSpec, RotaryAlgorithm, RotaryLayoutSpec, RotarySpec,
};
use serde::{Deserialize, Serialize};
pub use window::{AttentionWindow, QueryTemperature};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "snake_case")]
pub enum MixerSpec {
    SoftmaxAttention(AttentionSpec),
    LinearAttention(LinearAttentionSpec),
    StateSpace(StateSpaceSpec),
    LatentAttention(LatentAttentionSpec),
}

impl MixerSpec {
    /// The softmax attention this mixer runs, if any.
    #[must_use]
    pub const fn softmax_attention(&self) -> Option<&AttentionSpec> {
        match self {
            Self::SoftmaxAttention(attention) => Some(attention),
            _ => None,
        }
    }
}

/// Multi-head latent attention: keys and values are reconstructed per head
/// from one shared latent, and every head shares one rotary key, so a token
/// caches `key_value_rank + rope_head_dim` values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentAttentionSpec {
    pub query_heads: usize,
    /// Rank of the normalized query down-projection; absent when queries
    /// project directly from the hidden state.
    pub query_rank: Option<usize>,
    pub key_value_rank: usize,
    pub nope_head_dim: usize,
    pub rope_head_dim: usize,
    pub value_head_dim: usize,
    /// RMS normalization of the query latent and of the key/value latent.
    pub latent_normalization: super::NormalizationSpec,
    pub rotary_pairing: RotaryPairingSpec,
    pub scale: f64,
    /// Rotation of the `rope_head_dim` part; `None` for `NoPE` variants.
    pub position: PositionEncodingSpec,
    /// Position-dependent scale on every query; absent when all positions
    /// attend at `scale`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_temperature: Option<QueryTemperature>,
}

impl LatentAttentionSpec {
    /// Query/key width of one head.
    #[must_use]
    pub const fn query_head_dim(&self) -> usize {
        self.nope_head_dim + self.rope_head_dim
    }

    /// Values one token keeps in the cache.
    #[must_use]
    pub const fn cached_width(&self) -> usize {
        self.key_value_rank + self.rope_head_dim
    }
}

/// How the rotary rows of a projection pair up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotaryPairingSpec {
    /// Adjacent rows `(2i, 2i + 1)` rotate together.
    Interleaved,
    /// Rows `i` and `i + d/2` rotate together.
    Halves,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<AttentionWindow>,
    pub position: PositionEncodingSpec,
    /// Cap on scores before the softmax, `cap · tanh(s / cap)` (Gemma 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_softcap: Option<f64>,
    /// Position-dependent scale on every query (Llama 4's rotation-free
    /// layers); absent when all positions attend at `scale`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_temperature: Option<QueryTemperature>,
    /// The earlier layer whose keys and values this layer attends over
    /// instead of projecting its own (Gemma 4's E models); its own query
    /// rotates as that layer's keys do.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_value_source: Option<usize>,
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
    /// RMS norms without a learned scale on each query and key head, with the
    /// model's `rms_norm_eps` (Llama 4's rotary layers: transformers
    /// `Llama4TextL2Norm(config.rms_norm_eps)`, as llama.cpp).
    QueryKeyUnitRms,
    /// Learned RMS norms over each token's whole query and key projections,
    /// across heads (`OLMo` 2).
    QueryKeyRmsAcrossHeads,
    /// Learned RMS norms on queries and keys and an RMS norm without a learned
    /// scale on values. No checkpoint tensor shows the value norm, so only a
    /// model spec declares this variant.
    QueryKeyValueRms,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionOutputSpec {
    Direct,
    Gated,
}
