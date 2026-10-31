//! Typed configuration for decoders that interleave selective state-space
//! (Mamba-2) blocks with attention and routed-expert blocks.
//!
//! Checkpoints in this family describe one mixer or one feed-forward per
//! physical block. libmir composes a logical decoder layer from a mixer block
//! and the routed-expert block that directly follows it, so the layer shapes
//! match the mixer-plus-feed-forward composition used everywhere else.

use std::num::NonZeroUsize;

mod blocks;
mod parse;

pub use blocks::{HybridBlockKind, HybridBlockPattern, HybridLayer, HybridLayerBlock};
pub(super) use parse::state_space_hybrid;

/// Complete configuration of a state-space hybrid decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct StateSpaceHybridConfig {
    /// Physical block sequence and its logical layer pairing.
    pub blocks: HybridBlockPattern,
    /// Mamba-2 mixer geometry shared by every state-space block.
    pub mamba: Mamba2Config,
    /// Routed-expert geometry shared by every routed block.
    pub routed: LatentRoutedConfig,
    /// Multi-token prediction modules, when the checkpoint declares them.
    pub prediction: Option<PredictionModules>,
}

/// Multi-token prediction (MTP) modules after the decoder.
///
/// Each module fuses the embedding of the next token with the decoder's last
/// hidden state, then runs one attention block and one routed block with the
/// decoder's geometry to draft a further token. Physical module blocks are
/// `mtp.layers.{2d}` and `mtp.layers.{2d + 1}` for module `d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictionModules {
    /// Number of modules (`num_nextn_predict_layers`).
    pub depth: NonZeroUsize,
}

/// Mamba-2 selective state-space mixer geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct Mamba2Config {
    /// Number of SSM heads.
    pub heads: usize,
    /// Channels per head.
    pub head_dim: usize,
    /// Recurrent state width per head.
    pub state_size: usize,
    /// Number of shared B/C groups; heads are split evenly across groups.
    pub groups: usize,
    /// Causal depthwise convolution width.
    pub convolution_kernel_size: usize,
    /// Whether the convolution carries a per-channel bias.
    pub convolution_bias: bool,
    /// Whether the input and output projections carry a bias.
    pub projection_bias: bool,
    /// Chunk length of the structured state-space duality prefill.
    pub chunk_size: usize,
    /// Clamp applied to `softplus(dt + dt_bias)`.
    pub time_step_limit: TimeStepLimit,
    /// Element type of the persistent recurrent state.
    pub state_dtype: StateSpaceStateDType,
}

/// Lower and upper clamp of the discretization time step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeStepLimit {
    /// Inclusive lower bound.
    pub minimum: f64,
    /// Inclusive upper bound, or none when the time step is unbounded.
    pub maximum: TimeStepBound,
}

/// Upper bound of the discretization time step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeStepBound {
    /// The time step is clamped to this finite value.
    Finite(f64),
    /// The time step is not clamped from above.
    Unbounded,
}

/// Element type used to persist the recurrent state between steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpaceStateDType {
    /// IEEE single precision.
    Float32,
    /// bfloat16.
    BFloat16,
    /// IEEE half precision.
    Float16,
}

/// Routed-expert block that runs experts in a projected latent space.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentRoutedConfig {
    /// Number of routed experts.
    pub expert_count: usize,
    /// Experts selected per token.
    pub top_k: usize,
    /// Width of the latent space routed experts operate in.
    pub latent_size: usize,
    /// Intermediate width of one routed expert.
    pub expert_intermediate_size: usize,
    /// Intermediate width of the always-active shared expert, if present.
    pub shared_intermediate_size: Option<usize>,
    /// Multiplier applied to renormalized routing weights.
    pub routed_scale: f64,
}

impl Mamba2Config {
    /// Width of the gated SSM output (`heads * head_dim`).
    #[must_use]
    pub const fn inner_size(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Channels passing through the causal convolution (`x`, `B`, and `C`).
    #[must_use]
    pub const fn convolution_channels(&self) -> usize {
        self.inner_size() + 2 * self.groups * self.state_size
    }

    /// Width of the fused input projection (`z`, `xBC`, and `dt`).
    #[must_use]
    pub const fn input_projection_size(&self) -> usize {
        self.inner_size() + self.convolution_channels() + self.heads
    }

    /// Channels normalized together by the gated grouped `RMSNorm`.
    #[must_use]
    pub const fn norm_group_size(&self) -> usize {
        self.inner_size() / self.groups
    }
}

#[cfg(test)]
mod tests;
