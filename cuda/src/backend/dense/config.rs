use super::{
    Bf16LinearPairWeights, CudaTensor, DecodeAttentionConfig, DecodeAttentionWeights,
    routed::RoutedExpertWeights,
};
use crate::{
    AffineQuantizedWeight, BlockFp8LinearWeight, DirectFp8CheckpointWeight, Error,
    Fp8ResidualLinearWeight, GgmlCheckpointWeight, MxFp4CheckpointWeight, MxFp8CheckpointWeight,
    NvFp4LinearWeight, PackedIntegerWeight, Result,
    kernels::{RouterUnitScoring, RouterUnitWeighting},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseSwiGluConfig {
    pub attention: DecodeAttentionConfig,
    pub intermediate_size: usize,
    pub activation: crate::GatedActivation,
    /// Storage of the gate, up and down projections; attention projections
    /// follow `attention.projection_format`.
    pub feed_forward_format: crate::ProjectionFormat,
    /// Routed experts in place of the gated MLP; `intermediate_size` is then
    /// the width of one expert.
    pub experts: Option<DenseRoutedExperts>,
    pub feed_forward_input: FeedForwardInput,
}

/// The stream a dense layer's feed-forward normalizes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FeedForwardInput {
    /// The residual after attention (sequential blocks).
    #[default]
    AttentionResidual,
    /// The layer input that attention also reads (Cohere's parallel block,
    /// `x + attention(norm(x)) + feed_forward(norm(x))`).
    LayerInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedExpertsConfig {
    pub count: usize,
    pub top_k: usize,
}

/// A dense layer's routed experts and how their router weighs them.
///
/// Every weight takes a factor, no router limits expert groups, and a
/// sigmoid router ([`RouterUnitWeighting::SigmoidBiased`]) loads its layer's
/// correction bias.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseRoutedExperts {
    pub experts: RoutedExpertsConfig,
    pub weighting: RouterUnitWeighting,
    pub scale: f32,
    /// The width of an ungated shared expert beside the routed ones
    /// (GLM-4.5, Llama 4).
    pub shared_intermediate: Option<std::num::NonZeroU32>,
}

impl DenseRoutedExperts {
    #[must_use]
    pub const fn router(self) -> RouterUnitScoring {
        RouterUnitScoring {
            weighting: self.weighting,
            group_limit: None,
            scale: self.scale,
        }
    }
}

impl DenseSwiGluConfig {
    /// NVFP4 attention projections keep their layer's prepared weights, so
    /// such executors cannot be shared between layers. Feed-forward
    /// projections, a router's correction bias and a shared expert take each
    /// layer's weights at execution.
    #[must_use]
    pub const fn binds_layer_weights(&self) -> bool {
        matches!(self.attention.projection_format, crate::ProjectionFormat::NvFp4)
    }
}

#[derive(Clone, Copy)]
pub struct DenseSwiGluWeights<'a> {
    pub attention: DecodeAttentionWeights<'a>,
    /// Normalizes the residual stream before the feed-forward projections;
    /// `None` in post-norm layers, which normalize the sub-block outputs
    /// through `sandwich` instead.
    pub post_attention_norm: Option<&'a CudaTensor>,
    pub sandwich: Option<DenseSandwichWeights<'a>>,
    pub feed_forward: DenseFeedForwardWeights<'a>,
}

#[derive(Clone, Copy)]
pub enum DenseFeedForwardWeights<'a> {
    Gated {
        gate_up: DenseGateUpWeights<'a>,
        down: DenseDownWeight<'a>,
    },
    Routed(&'a RoutedExpertWeights),
}

/// Gemma-style norms on each sub-block's output before its residual add, and
/// a learned BF16 scalar on the layer output when the checkpoint has one.
#[derive(Clone, Copy)]
pub struct DenseSandwichWeights<'a> {
    pub attention_output_norm: &'a CudaTensor,
    pub feed_forward_output_norm: &'a CudaTensor,
    pub layer_scalar: Option<&'a CudaTensor>,
}

#[derive(Clone, Copy)]
pub enum DenseGateUpWeights<'a> {
    Affine {
        gate: &'a AffineQuantizedWeight,
        up: &'a AffineQuantizedWeight,
    },
    Bf16(&'a Bf16LinearPairWeights),
    DirectFp8 {
        gate: &'a DirectFp8CheckpointWeight,
        up: &'a DirectFp8CheckpointWeight,
    },
    Ggml {
        gate: &'a GgmlCheckpointWeight,
        up: &'a GgmlCheckpointWeight,
    },
    MxFp4 {
        gate: &'a MxFp4CheckpointWeight,
        up: &'a MxFp4CheckpointWeight,
    },
    MxFp8 {
        gate: &'a MxFp8CheckpointWeight,
        up: &'a MxFp8CheckpointWeight,
    },
    PackedInteger {
        gate: &'a PackedIntegerWeight,
        up: &'a PackedIntegerWeight,
    },
    NvFp4 {
        gate: &'a NvFp4LinearWeight,
        up: &'a NvFp4LinearWeight,
    },
    BlockFp8 {
        exact: &'a Bf16LinearPairWeights,
        quantized: &'a BlockFp8LinearWeight,
    },
    Fp8Int4 {
        exact: &'a Bf16LinearPairWeights,
        quantized: &'a Fp8ResidualLinearWeight,
    },
}

impl<'a> DenseGateUpWeights<'a> {
    pub(super) fn require_bf16(self) -> Result<&'a Bf16LinearPairWeights> {
        match self {
            Self::Bf16(weights)
            | Self::BlockFp8 { exact: weights, .. }
            | Self::Fp8Int4 { exact: weights, .. } => Ok(weights),
            Self::Affine { .. }
            | Self::DirectFp8 { .. }
            | Self::Ggml { .. }
            | Self::MxFp4 { .. }
            | Self::MxFp8 { .. }
            | Self::PackedInteger { .. }
            | Self::NvFp4 { .. } => {
                Err(Error::InvalidExecutionPlan("BF16 gate/up operation received NVFP4 weights"))
            },
        }
    }
}

#[derive(Clone, Copy)]
pub enum DenseDownWeight<'a> {
    Affine(&'a AffineQuantizedWeight),
    Bf16(&'a CudaTensor),
    DirectFp8(&'a DirectFp8CheckpointWeight),
    Ggml(&'a GgmlCheckpointWeight),
    MxFp4(&'a MxFp4CheckpointWeight),
    MxFp8(&'a MxFp8CheckpointWeight),
    PackedInteger(&'a PackedIntegerWeight),
    NvFp4(&'a NvFp4LinearWeight),
    BlockFp8 {
        exact: &'a CudaTensor,
        quantized: &'a BlockFp8LinearWeight,
    },
    Fp8Int4 {
        exact: &'a CudaTensor,
        quantized: &'a Fp8ResidualLinearWeight,
    },
}

impl<'a> DenseDownWeight<'a> {
    pub(super) fn require_bf16(self) -> Result<&'a CudaTensor> {
        match self {
            Self::Bf16(weight)
            | Self::BlockFp8 { exact: weight, .. }
            | Self::Fp8Int4 { exact: weight, .. } => Ok(weight),
            Self::Affine(_)
            | Self::DirectFp8(_)
            | Self::Ggml(_)
            | Self::MxFp4(_)
            | Self::MxFp8(_)
            | Self::PackedInteger(_)
            | Self::NvFp4(_) => {
                Err(Error::InvalidExecutionPlan("BF16 down operation received NVFP4 weight"))
            },
        }
    }
}
