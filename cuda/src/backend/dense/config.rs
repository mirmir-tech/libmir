use super::{Bf16LinearPairWeights, CudaTensor, DecodeAttentionConfig, DecodeAttentionWeights};
use crate::{
    AffineQuantizedWeight, BlockFp8LinearWeight, DirectFp8CheckpointWeight, Error,
    Fp8ResidualLinearWeight, MxFp4CheckpointWeight, MxFp8CheckpointWeight, NvFp4LinearWeight,
    PackedIntegerWeight, Result,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseSwiGluConfig {
    pub attention: DecodeAttentionConfig,
    pub intermediate_size: usize,
    pub activation: crate::GatedActivation,
}

#[derive(Clone, Copy)]
pub struct DenseSwiGluWeights<'a> {
    pub attention: DecodeAttentionWeights<'a>,
    pub post_attention_norm: &'a CudaTensor,
    pub gate_up: DenseGateUpWeights<'a>,
    pub down: DenseDownWeight<'a>,
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
            | Self::MxFp4(_)
            | Self::MxFp8(_)
            | Self::PackedInteger(_)
            | Self::NvFp4(_) => {
                Err(Error::InvalidExecutionPlan("BF16 down operation received NVFP4 weight"))
            },
        }
    }
}
