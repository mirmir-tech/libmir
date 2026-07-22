use super::{Bf16LinearPairWeights, CudaTensor, DecodeAttentionConfig, DecodeAttentionWeights};
use crate::{CompressedInt8Weight, Error, NvFp4LinearWeight, Result};

#[derive(Clone, Copy, Debug)]
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
    Bf16(&'a Bf16LinearPairWeights),
    Int8 {
        gate: &'a CompressedInt8Weight,
        up: &'a CompressedInt8Weight,
    },
    NvFp4 {
        gate: &'a NvFp4LinearWeight,
        up: &'a NvFp4LinearWeight,
    },
}

impl<'a> DenseGateUpWeights<'a> {
    pub(super) fn require_bf16(self) -> Result<&'a Bf16LinearPairWeights> {
        match self {
            Self::Bf16(weights) => Ok(weights),
            Self::Int8 { .. } | Self::NvFp4 { .. } => {
                Err(Error::InvalidExecutionPlan("BF16 gate/up operation received NVFP4 weights"))
            },
        }
    }
}

#[derive(Clone, Copy)]
pub enum DenseDownWeight<'a> {
    Bf16(&'a CudaTensor),
    Int8(&'a CompressedInt8Weight),
    NvFp4(&'a NvFp4LinearWeight),
}

impl<'a> DenseDownWeight<'a> {
    pub(super) fn require_bf16(self) -> Result<&'a CudaTensor> {
        match self {
            Self::Bf16(weight) => Ok(weight),
            Self::Int8(_) | Self::NvFp4(_) => {
                Err(Error::InvalidExecutionPlan("BF16 down operation received NVFP4 weight"))
            },
        }
    }
}
