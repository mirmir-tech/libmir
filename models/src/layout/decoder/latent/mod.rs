//! Multi-head latent attention: queries and keys rebuilt from shared latents.

mod parse;

use std::num::NonZeroUsize;

pub(super) use parse::latent_decoder;

use crate::semantic::QueryTemperature;

/// A decoder whose attention layers all use multi-head latent attention.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentDecoderConfig {
    pub attention: LatentAttentionConfig,
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
