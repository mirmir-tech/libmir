use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum LogicalTensorRole {
    Embedding,
    FinalNorm,
    Output,
    VisionProjection,
    Layer {
        index: usize,
        tensor: LayerTensorRole,
    },
    /// A tensor of multi-token prediction module `depth`.
    Prediction {
        depth: usize,
        tensor: PredictionTensorRole,
    },
    Auxiliary {
        path: String,
    },
}

/// Tensors of one multi-token prediction module.
///
/// The fusion of the next token's embedding with the decoder's last hidden
/// state, then an attention block and a routed block with decoder-layer
/// roles, then the output norm before the shared head.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PredictionTensorRole {
    /// `enorm`: normalizes the embedding of the next token.
    EmbeddingNorm,
    /// `hnorm`: normalizes the decoder's last hidden state.
    HiddenNorm,
    /// `eh_proj`: projects the concatenated pair back to the hidden width.
    Fusion,
    /// `final_layernorm`: normalizes the module output before the shared head.
    FinalNorm,
    /// `k_scale` / `v_scale`: FP8 K/V cache scales of the module's attention,
    /// unused while K/V stays in BF16.
    CacheScale { projection: AttentionProjectionRole },
    /// A tensor of the module's attention or routed block.
    Layer { tensor: LayerTensorRole },
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
mod prediction;

pub(super) use parse::parse;
