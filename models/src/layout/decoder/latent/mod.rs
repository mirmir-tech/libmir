//! Multi-head latent attention and the mixture-of-experts layout paired with
//! it: leading dense layers, then routed plus shared experts.

mod parse;

use std::num::NonZeroUsize;

pub(super) use parse::latent_decoder;

use crate::semantic::{ExpertGroupLimit, QueryTemperature};

/// A decoder whose attention layers all use multi-head latent attention.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentDecoderConfig {
    pub attention: LatentAttentionConfig,
    /// Absent when every layer has a dense feed-forward.
    pub experts: Option<LatentExpertsConfig>,
}

/// Queries and keys split into a position-free part reconstructed from a
/// shared latent and a rotary part shared by all heads; the cache keeps only
/// the latent and the rotary key of every token.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentAttentionConfig {
    /// Rank of the query down-projection; absent when queries project directly.
    pub query_rank: Option<NonZeroUsize>,
    /// Width of the key/value latent each token caches.
    pub key_value_rank: usize,
    /// Per-head width of the position-free query/key part.
    pub nope_head_dim: usize,
    /// Width of the rotary query/key part; the key part is shared by heads.
    pub rope_head_dim: usize,
    pub value_head_dim: usize,
    pub rotary_pairing: RotaryPairing,
    /// Softmax scale, including `YaRN`'s `mscale_all_dim` correction.
    pub scale: f64,
    /// `llama_4_scaling_beta` over `original_max_position_embeddings`.
    pub query_temperature: Option<QueryTemperature>,
}

impl LatentAttentionConfig {
    /// Query/key width of one head.
    #[must_use]
    pub const fn query_head_dim(&self) -> usize {
        self.nope_head_dim + self.rope_head_dim
    }

    /// Values one token keeps in the cache: its latent and its rotary key.
    #[must_use]
    pub const fn cached_width(&self) -> usize {
        self.key_value_rank + self.rope_head_dim
    }
}

/// How the rotary dimensions of a checkpoint's projection rows pair up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryPairing {
    /// Adjacent rows `(2i, 2i + 1)` rotate together; `DeepSeek`'s reference
    /// code permutes them into halves before rotating.
    Interleaved,
    /// Rows `i` and `i + d/2` rotate together (`rotate_half`).
    Halves,
}

/// Routed feed-forward with leading dense layers: the first layers stay
/// dense, the others combine routed experts with optional always-active
/// shared experts.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentExpertsConfig {
    pub first_dense_layers: usize,
    /// Shared experts, fused into one feed-forward of `count` times the expert
    /// width.
    pub shared_experts: Option<NonZeroUsize>,
    pub routing: ExpertRouting,
    /// Groups the router selects experts from, when more than one.
    pub group_limit: Option<ExpertGroupLimit>,
    /// Factor on the combined routed-expert output (`routed_scaling_factor`).
    pub output_scale: f64,
}

/// How a router scores experts and weighs the ones it selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRouting {
    /// Select by softmax probability over every expert; `renormalize` makes
    /// the selected weights sum to one.
    Softmax { renormalize: bool },
    /// Select by `sigmoid(logit) + correction_bias`, weigh by the
    /// renormalized `sigmoid(logit)` of the selected experts.
    SigmoidWithCorrectionBias,
}
