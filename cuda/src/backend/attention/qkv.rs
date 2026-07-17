use mircuda::{DeviceBuffer, bf16};

use super::{
    Bf16LinearPack, CudaBackend, DecodeAttentionConfig, DecodeQkvWeights, ProjectionFormat,
};
use crate::{CudaTensor, Error, ExecutionPhase, NvFp4Bf16Pack, Result, RmsNormBf16};

#[derive(Debug)]
pub(super) enum AttentionQkvProjection {
    Bf16(Bf16LinearPack<3>),
    NvFp4(Box<NvFp4Bf16Pack<3>>),
}

pub(super) struct QkvProjectionBuffers<'a> {
    pub(super) normalized: &'a mut DeviceBuffer<bf16>,
    pub(super) packed: &'a mut DeviceBuffer<bf16>,
    pub(super) separate: &'a mut [DeviceBuffer<bf16>; 3],
}

impl AttentionQkvProjection {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        tokens: usize,
        weights: Option<DecodeQkvWeights<'_>>,
    ) -> Result<Self> {
        let key = config.cache.kv_heads * config.cache.key_head_dim;
        let value = config.cache.kv_heads * config.cache.value_head_dim;
        let query = config.query_heads * config.cache.key_head_dim;
        match config.projection_format {
            ProjectionFormat::Bf16 => Ok(Self::Bf16(Bf16LinearPack::new(
                backend,
                if tokens == 1 {
                    ExecutionPhase::Decode
                } else {
                    ExecutionPhase::Prefill
                },
                tokens,
                config.hidden_size,
                [query, key, value],
            )?)),
            ProjectionFormat::NvFp4 => {
                let DecodeQkvWeights::NvFp4(weights) = weights.ok_or(
                    Error::InvalidExecutionPlan("NVFP4 attention requires prepared QKV weights"),
                )?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "NVFP4 attention received non-NVFP4 QKV weights",
                    ));
                };
                Ok(Self::NvFp4(Box::new(NvFp4Bf16Pack::from_weights(
                    backend,
                    tokens,
                    weights.map(Clone::clone),
                )?)))
            },
        }
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        input_norm: &RmsNormBf16,
        norm_weight: &CudaTensor,
        weights: DecodeQkvWeights<'_>,
        buffers: &mut QkvProjectionBuffers<'_>,
    ) -> Result<bool> {
        match self {
            Self::Bf16(operation) => {
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                operation.execute(buffers.normalized, weights.require_bf16()?, buffers.packed)?;
                Ok(false)
            },
            Self::NvFp4(operation) => {
                let DecodeQkvWeights::NvFp4(_) = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "NVFP4 QKV operation received BF16 weights",
                    ));
                };
                operation.execute_rms_norm(
                    input,
                    input_norm.weight(norm_weight)?,
                    input_norm.epsilon(),
                    buffers.separate,
                )?;
                Ok(true)
            },
        }
    }
}
