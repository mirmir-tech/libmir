mod load;

use super::projection::BoundLinear;
use crate::engine::{Array, NormWeight, Result, Stream};

#[derive(Debug)]
pub(super) struct AttentionWeights {
    pub(super) query: BoundLinear,
    pub(super) key: BoundLinear,
    pub(super) value: BoundLinear,
    pub(super) output: BoundLinear,
    pub(super) qk_norm: QkNorm,
    pub(super) rope_frequencies: Option<Array>,
}

/// How queries and keys are normalized before the rotation.
#[derive(Debug)]
pub(super) enum QkNorm {
    None,
    /// Each head by a learned scale.
    Head {
        query: NormWeight,
        key: NormWeight,
    },
    /// The whole projection before the head split (`OLMo` 2).
    Projection {
        query: NormWeight,
        key: NormWeight,
    },
    /// Each head without a learned scale (Llama 4).
    UnitHead,
}

/// Which projection a Q/K norm applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QkSide {
    Query,
    Key,
}

impl QkNorm {
    pub(super) const fn weight(&self, side: QkSide) -> Option<&NormWeight> {
        match self {
            Self::Head { query, key } | Self::Projection { query, key } => Some(match side {
                QkSide::Query => query,
                QkSide::Key => key,
            }),
            Self::None | Self::UnitHead => None,
        }
    }
}

#[derive(Debug)]
pub(super) struct MlpWeights {
    pub(super) gate: BoundLinear,
    pub(super) up: BoundLinear,
    pub(super) down: BoundLinear,
}

/// The layer's feed-forward: a gated MLP, or routed experts in its place.
#[derive(Debug)]
pub(super) enum FeedForwardWeights {
    Gated(Box<MlpWeights>),
    /// Routed experts, with a shared expert beside them for Qwen2-MoE and
    /// GLM-4.5.
    Routed(Box<crate::engine::SharedExpertMoe>),
}

/// Gemma-style norms on each sub-block's output before its residual add.
#[derive(Debug)]
pub(super) struct SandwichNorms {
    pub(super) attention_output: NormWeight,
    pub(super) feed_forward_output: NormWeight,
}

/// Where the layer normalizes: before each sub-block (with Gemma's output
/// norms when present), or only on each sub-block's output (`OLMo` 2).
#[derive(Debug)]
pub(super) enum LayerNorms {
    Pre {
        input: NormWeight,
        feed_forward: NormWeight,
        output: Option<SandwichNorms>,
    },
    Post(SandwichNorms),
    /// One norm feeds attention and the feed-forward, which read the same
    /// stream (Cohere).
    Parallel {
        input: NormWeight,
    },
}

#[derive(Debug)]
pub(super) struct DenseSwiGluWeights {
    pub(super) norms: LayerNorms,
    pub(super) attention: AttentionWeights,
    pub(super) feed_forward: FeedForwardWeights,
}

impl DenseSwiGluWeights {
    /// The attention input: the normalized stream, or the stream itself.
    pub(super) fn attention_input(
        &self,
        input: &Array,
        eps: f32,
        stream: &Stream,
    ) -> Result<Array> {
        match &self.norms {
            LayerNorms::Pre { input: norm, .. } | LayerNorms::Parallel { input: norm } => {
                norm.apply(input, eps, stream)
            },
            LayerNorms::Post(_) => Array::from_native(input.native().clone()),
        }
    }

    /// The feed-forward input: the normalized stream, or the stream itself.
    pub(super) fn feed_forward_input(
        &self,
        input: &Array,
        eps: f32,
        stream: &Stream,
    ) -> Result<Array> {
        match &self.norms {
            LayerNorms::Pre { feed_forward, .. } => feed_forward.apply(input, eps, stream),
            LayerNorms::Parallel { input: norm } => norm.apply(input, eps, stream),
            LayerNorms::Post(_) => Array::from_native(input.native().clone()),
        }
    }

    /// Norms on each sub-block's output before its residual add.
    pub(super) const fn output_norms(&self) -> Option<&SandwichNorms> {
        match &self.norms {
            LayerNorms::Pre { output, .. } => output.as_ref(),
            LayerNorms::Post(output) => Some(output),
            LayerNorms::Parallel { .. } => None,
        }
    }

    /// Whether attention and the feed-forward read the same stream.
    pub(super) const fn parallel(&self) -> bool {
        matches!(self.norms, LayerNorms::Parallel { .. })
    }
}
