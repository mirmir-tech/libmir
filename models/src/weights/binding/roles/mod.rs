use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum LogicalTensorRole {
    Embedding,
    FinalNorm,
    Output,
    VisionProjection,
    Layer { index: usize, tensor: LayerTensorRole },
    Auxiliary { path: String },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayerTensorRole {
    InputNorm,
    PostAttentionNorm,
    PreDenseNorm,
    PostDenseNorm,
    PreExpertNorm,
    PostExpertNorm,
    PostFeedForwardNorm,
    AttentionProjection {
        projection: AttentionProjectionRole,
    },
    QueryNorm,
    KeyNorm,
    AttentionSinks,
    LinearAttention {
        tensor: LinearAttentionTensorRole,
    },
    Router,
    RouterNormScale,
    RouterExpertScale,
    LayerScale,
    FeedForwardProjection {
        projection: FeedForwardProjectionRole,
    },
    ExpertProjection {
        expert: Option<usize>,
        projection: ExpertProjectionRole,
    },
    SharedExpertProjection {
        projection: FeedForwardProjectionRole,
    },
    SharedExpertOutputGate,
    StateSpace {
        tensor: StateSpaceTensorRole,
    },
    RouterCorrectionBias,
    LatentProjection {
        projection: LatentProjectionRole,
    },
    Auxiliary {
        path: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionProjectionRole {
    Query,
    Key,
    Value,
    Qkv,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedForwardProjectionRole {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertProjectionRole {
    Gate,
    Up,
    GateUp,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearAttentionTensorRole {
    DecayLog,
    Convolution,
    TimeBias,
    QkvProjection,
    GateProjection,
    AlphaProjection,
    BetaProjection,
    Norm,
    OutputProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateSpaceTensorRole {
    /// `A_log`: logarithm of the negated per-head decay.
    DecayLog,
    /// `D`: per-head skip connection scale.
    SkipScale,
    /// Causal depthwise convolution over `x`, `B`, and `C`.
    Convolution,
    /// Per-head time-step bias.
    TimeBias,
    /// Fused `z | xBC | dt` input projection.
    InputProjection,
    /// Gated grouped `RMSNorm` weight.
    GatedNorm,
    OutputProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentProjectionRole {
    /// Hidden state to the latent space routed experts run in.
    Input,
    /// Latent expert output back to the hidden state.
    Output,
}

mod paired;
mod parse;

pub(super) use parse::parse;
