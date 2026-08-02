use runtime::kv::KvStorageSpec;

use super::{
    Bf16LinearPackWeights, CudaTensor, DecodeAttentionOutputWeight, DeviceBuffer,
    NvFp4LinearWeight, ProjectionFormat, bf16,
};
use crate::{
    AffineQuantizedWeight, DirectFp8CheckpointWeight, Error, MxFp4CheckpointWeight,
    MxFp8CheckpointWeight, PackedIntegerWeight, Result,
};

/// Config-derived geometry for one fixed-shape decode attention layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeAttentionConfig {
    pub layer: usize,
    pub hidden_size: usize,
    pub query_heads: usize,
    pub rotary_dim: usize,
    pub rope_pairing_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_epsilon: f32,
    pub attention_scale: f32,
    pub projection_format: ProjectionFormat,
    pub qkv_normalization: crate::kernels::QkvNormalization,
    pub sliding_window: Option<usize>,
    pub max_sequence_blocks: usize,
    pub cache: KvStorageSpec,
}

/// Checkpoint tensors consumed by generic decode attention.
#[derive(Clone, Copy)]
pub struct DecodeAttentionWeights<'a> {
    pub input_norm: &'a CudaTensor,
    pub qkv: DecodeQkvWeights<'a>,
    pub query_norm: &'a DeviceBuffer<bf16>,
    pub key_norm: &'a DeviceBuffer<bf16>,
    pub output: DecodeAttentionOutputWeight<'a>,
}

#[derive(Clone, Copy)]
pub enum DecodeQkvWeights<'a> {
    Affine([&'a AffineQuantizedWeight; 3]),
    Bf16(&'a Bf16LinearPackWeights<3>),
    DirectFp8([&'a DirectFp8CheckpointWeight; 3]),
    MxFp4([&'a MxFp4CheckpointWeight; 3]),
    MxFp8([&'a MxFp8CheckpointWeight; 3]),
    PackedInteger([&'a PackedIntegerWeight; 3]),
    NvFp4([&'a NvFp4LinearWeight; 3]),
}

impl<'a> DecodeQkvWeights<'a> {
    pub(in crate::backend) fn require_bf16(self) -> Result<&'a Bf16LinearPackWeights<3>> {
        match self {
            Self::Bf16(weights) => Ok(weights),
            Self::Affine(_)
            | Self::DirectFp8(_)
            | Self::MxFp4(_)
            | Self::MxFp8(_)
            | Self::PackedInteger(_)
            | Self::NvFp4(_) => {
                Err(Error::InvalidExecutionPlan("BF16 QKV operation received NVFP4 weights"))
            },
        }
    }
}

pub(super) fn validate(config: DecodeAttentionConfig) -> Result<()> {
    if config.hidden_size == 0
        || config.query_heads == 0
        || config.max_sequence_blocks == 0
        || config.cache.kv_heads == 0
        || !config.query_heads.is_multiple_of(config.cache.kv_heads)
        || !config.attention_scale.is_finite()
    {
        Err(Error::InvalidPagedKv("invalid decode attention configuration"))
    } else {
        Ok(())
    }
}
