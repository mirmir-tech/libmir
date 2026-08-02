use mircuda::{DeviceBuffer, bf16};

use super::{
    Bf16LinearPack, CudaBackend, DecodeAttentionConfig, DecodeQkvWeights, ProjectionFormat,
};
use crate::{
    CudaTensor, DirectFp8Bf16Linear, Error, ExecutionPhase, MxFp4Bf16Linear, MxFp8Bf16Linear,
    NvFp4Bf16Pack, PackedIntegerBf16Linear, Result, RmsNormBf16, backend::linear::AffineProjection,
};

#[derive(Debug)]
pub(super) enum AttentionQkvProjection {
    Affine(Box<[AffineProjection; 3]>),
    Bf16(Bf16LinearPack<3>),
    DirectFp8(Box<[DirectFp8Bf16Linear; 3]>),
    MxFp4(Box<[MxFp4Bf16Linear; 3]>),
    MxFp8(Box<[MxFp8Bf16Linear; 3]>),
    PackedInteger(Box<[PackedIntegerBf16Linear; 3]>),
    NvFp4(Box<NvFp4Bf16Pack<3>>),
}

pub(super) struct QkvProjectionBuffers<'a> {
    pub(super) normalized: &'a mut DeviceBuffer<bf16>,
    pub(super) packed: &'a mut DeviceBuffer<bf16>,
    pub(super) separate: &'a mut [DeviceBuffer<bf16>; 3],
}

impl AttentionQkvProjection {
    #[allow(clippy::too_many_lines)]
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
            ProjectionFormat::Affine => {
                let DecodeQkvWeights::Affine(weights) = weights.ok_or(
                    Error::InvalidExecutionPlan("affine attention requires prepared QKV weights"),
                )?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "affine attention received non-affine QKV weights",
                    ));
                };
                Ok(Self::Affine(Box::new([
                    affine(backend, tokens, config.hidden_size, query, weights[0])?,
                    affine(backend, tokens, config.hidden_size, key, weights[1])?,
                    affine(backend, tokens, config.hidden_size, value, weights[2])?,
                ])))
            },
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
            ProjectionFormat::DirectFp8 => {
                let DecodeQkvWeights::DirectFp8(weights) = weights.ok_or(
                    Error::InvalidExecutionPlan("direct FP8 attention requires prepared QKV"),
                )?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "direct FP8 attention received other QKV weights",
                    ));
                };
                Ok(Self::DirectFp8(Box::new([
                    weights[0].prepare(backend, tokens)?,
                    weights[1].prepare(backend, tokens)?,
                    weights[2].prepare(backend, tokens)?,
                ])))
            },
            ProjectionFormat::MxFp8 => {
                Ok(Self::MxFp8(super::mxfp8::prepare_qkv(backend, tokens, weights)?))
            },
            ProjectionFormat::MxFp4 => {
                Ok(Self::MxFp4(super::mxfp4::prepare_qkv(backend, tokens, weights)?))
            },
            ProjectionFormat::PackedInteger => {
                let DecodeQkvWeights::PackedInteger(weights) =
                    weights.ok_or(Error::InvalidExecutionPlan(
                        "packed integer attention requires prepared QKV weights",
                    ))?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "packed integer attention received other QKV weights",
                    ));
                };
                Ok(Self::PackedInteger(Box::new([
                    PackedIntegerBf16Linear::new(
                        backend,
                        tokens,
                        config.hidden_size,
                        query,
                        weights[0],
                    )?,
                    PackedIntegerBf16Linear::new(
                        backend,
                        tokens,
                        config.hidden_size,
                        key,
                        weights[1],
                    )?,
                    PackedIntegerBf16Linear::new(
                        backend,
                        tokens,
                        config.hidden_size,
                        value,
                        weights[2],
                    )?,
                ])))
            },
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
            Self::Affine(operations) => {
                let DecodeQkvWeights::Affine(weights) = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "affine QKV operation received non-affine weights",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                for ((operation, weight), output) in
                    operations.iter().zip(weights).zip(buffers.separate.iter_mut())
                {
                    operation.execute(buffers.normalized, weight, output)?;
                }
                Ok(true)
            },
            Self::Bf16(operation) => {
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                operation.execute(buffers.normalized, weights.require_bf16()?, buffers.packed)?;
                Ok(false)
            },
            Self::DirectFp8(operations) => {
                let DecodeQkvWeights::DirectFp8(weights) = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "direct FP8 QKV operation received other weights",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                for ((operation, weight), output) in
                    operations.iter().zip(weights).zip(buffers.separate.iter_mut())
                {
                    operation.execute(buffers.normalized, weight, output)?;
                }
                Ok(true)
            },
            Self::MxFp8(operations) => {
                let DecodeQkvWeights::MxFp8(weights) = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "MXFP8 QKV operation received other weights",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                for ((operation, weight), output) in
                    operations.iter().zip(weights).zip(buffers.separate.iter_mut())
                {
                    operation.execute(buffers.normalized, weight, output)?;
                }
                Ok(true)
            },
            Self::MxFp4(operations) => {
                let DecodeQkvWeights::MxFp4(weights) = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "MXFP4 QKV operation received other weights",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                for ((operation, weight), output) in
                    operations.iter().zip(weights).zip(buffers.separate.iter_mut())
                {
                    operation.execute(buffers.normalized, weight, output)?;
                }
                Ok(true)
            },
            Self::PackedInteger(operations) => {
                let DecodeQkvWeights::PackedInteger(weights) = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "packed integer QKV operation received other weights",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                for ((operation, weight), output) in
                    operations.iter().zip(weights).zip(buffers.separate.iter_mut())
                {
                    operation.execute(buffers.normalized, weight, output)?;
                }
                Ok(true)
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

fn affine(
    backend: &CudaBackend,
    tokens: usize,
    input: usize,
    output: usize,
    weight: &crate::AffineQuantizedWeight,
) -> Result<AffineProjection> {
    let config = weight.infer_config(1, input, output)?;
    AffineProjection::new(backend, tokens, input, output, config.group_size, config.bits, weight)
}
