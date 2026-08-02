use mircuda::{DeviceBuffer, Stream, bf16};

use super::{Bf16Projection, CudaBackend, DecodeAttentionConfig, ProjectionFormat};
use crate::{
    AffineQuantizedWeight, BlockFp8LinearWeight, CudaTensor, DenseExecution, DensePlanRequest,
    DenseRole, DirectFp8Bf16Linear, DirectFp8CheckpointWeight, Error, ExecutionPhase,
    Fp8ResidualLinearWeight, MxFp4Bf16Linear, MxFp4CheckpointWeight, MxFp8Bf16Linear,
    MxFp8CheckpointWeight, NvFp4Bf16Linear, NvFp4LinearWeight, PackedIntegerBf16Linear,
    PackedIntegerWeight, Result,
};

#[derive(Clone, Copy)]
pub enum DecodeAttentionOutputWeight<'a> {
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

impl<'a> DecodeAttentionOutputWeight<'a> {
    #[must_use]
    pub const fn bf16(self) -> Option<&'a CudaTensor> {
        match self {
            Self::Bf16(weight)
            | Self::BlockFp8 { exact: weight, .. }
            | Self::Fp8Int4 { exact: weight, .. } => Some(weight),
            Self::Affine(_)
            | Self::DirectFp8(_)
            | Self::MxFp4(_)
            | Self::MxFp8(_)
            | Self::PackedInteger(_)
            | Self::NvFp4(_) => None,
        }
    }
}

#[derive(Debug)]
pub(super) enum AttentionOutputProjection {
    Affine(crate::backend::linear::AffineProjection),
    Bf16(Bf16Projection),
    DirectFp8(DirectFp8Bf16Linear),
    MxFp4(MxFp4Bf16Linear),
    MxFp8(MxFp8Bf16Linear),
    PackedInteger(PackedIntegerBf16Linear),
    NvFp4(NvFp4Bf16Linear),
    BlockFp8,
    Fp8Int4,
}

impl AttentionOutputProjection {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        tokens: usize,
        weight: Option<DecodeAttentionOutputWeight<'_>>,
    ) -> Result<Self> {
        let input_features = config.query_heads * config.cache.value_head_dim;
        let request = DensePlanRequest {
            phase: if tokens == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            role: DenseRole::AttentionOutput,
            tokens,
            input_features,
            output_features: config.hidden_size,
        };
        match config.projection_format {
            ProjectionFormat::Affine => {
                let Some(DecodeAttentionOutputWeight::Affine(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "affine attention requires affine output weight",
                    ));
                };
                let affine = weight.infer_config(1, input_features, config.hidden_size)?;
                Ok(Self::Affine(crate::backend::linear::AffineProjection::new(
                    backend,
                    tokens,
                    input_features,
                    config.hidden_size,
                    affine.group_size,
                    affine.bits,
                    weight,
                )?))
            },
            ProjectionFormat::PackedInteger => {
                let Some(DecodeAttentionOutputWeight::PackedInteger(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "packed integer attention requires prepared output weight",
                    ));
                };
                Ok(Self::PackedInteger(PackedIntegerBf16Linear::new(
                    backend,
                    tokens,
                    input_features,
                    config.hidden_size,
                    weight,
                )?))
            },
            ProjectionFormat::DirectFp8 => {
                let Some(DecodeAttentionOutputWeight::DirectFp8(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "direct FP8 attention requires prepared output weight",
                    ));
                };
                Ok(Self::DirectFp8(weight.prepare(backend, tokens)?))
            },
            ProjectionFormat::MxFp4 => {
                let Some(DecodeAttentionOutputWeight::MxFp4(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "MXFP4 attention requires prepared output weight",
                    ));
                };
                Ok(Self::MxFp4(weight.prepare(backend, tokens)?))
            },
            ProjectionFormat::MxFp8 => {
                let Some(DecodeAttentionOutputWeight::MxFp8(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "MXFP8 attention requires prepared output weight",
                    ));
                };
                Ok(Self::MxFp8(weight.prepare(backend, tokens)?))
            },
            ProjectionFormat::NvFp4 => {
                let Some(DecodeAttentionOutputWeight::NvFp4(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "NVFP4 attention requires prepared output weight",
                    ));
                };
                Ok(Self::NvFp4(NvFp4Bf16Linear::from_weight(backend, tokens, weight.clone())?))
            },
            ProjectionFormat::Bf16 => Ok(
                match backend
                    .execution_planner()
                    .plan_dense_with_prepared_weights(request)?
                    .execution()
                {
                    DenseExecution::BlockFp8Vector => Self::BlockFp8,
                    DenseExecution::Fp8Int4Vector => Self::Fp8Int4,
                    DenseExecution::Matrix | DenseExecution::Vector | DenseExecution::CublasLt => {
                        Self::Bf16(backend.prepare_bf16_projection(request)?)
                    },
                },
            ),
        }
    }

    pub(super) fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: DecodeAttentionOutputWeight<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match (self, weight) {
            (Self::Affine(operation), DecodeAttentionOutputWeight::Affine(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::Affine(_), _) => {
                Err(Error::InvalidExecutionPlan("affine attention output received other weight"))
            },
            (
                Self::PackedInteger(operation),
                DecodeAttentionOutputWeight::PackedInteger(weight),
            ) => operation.execute(input, weight, output),
            (Self::PackedInteger(_), _) => Err(Error::InvalidExecutionPlan(
                "packed integer attention output received other weight",
            )),
            (Self::NvFp4(operation), DecodeAttentionOutputWeight::NvFp4(_)) => {
                operation.execute(input, output)
            },
            (Self::NvFp4(_), _) => {
                Err(Error::InvalidExecutionPlan("NVFP4 attention output received non-NVFP4 weight"))
            },
            (Self::Bf16(operation), weight) => operation.execute(
                input,
                weight
                    .bf16()
                    .ok_or(Error::InvalidExecutionPlan("BF16 attention lacks BF16 weight"))?,
                output,
            ),
            (Self::DirectFp8(operation), DecodeAttentionOutputWeight::DirectFp8(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::DirectFp8(_), _) => Err(Error::InvalidExecutionPlan(
                "direct FP8 attention output received other weight",
            )),
            (Self::MxFp4(operation), DecodeAttentionOutputWeight::MxFp4(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::MxFp4(_), _) => {
                Err(Error::InvalidExecutionPlan("MXFP4 attention output received other weight"))
            },
            (Self::MxFp8(operation), DecodeAttentionOutputWeight::MxFp8(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::MxFp8(_), _) => {
                Err(Error::InvalidExecutionPlan("MXFP8 attention output received other weight"))
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::BlockFp8 { quantized, .. }) => {
                quantized.execute(stream, input, output)
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::Bf16(_)) => {
                Err(Error::InvalidExecutionPlan("block FP8 attention plan lacks quantized weight"))
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::DirectFp8(_)) => Err(
                Error::InvalidExecutionPlan("block FP8 attention plan received direct FP8 weight"),
            ),
            (Self::BlockFp8, DecodeAttentionOutputWeight::MxFp8(_)) => {
                Err(Error::InvalidExecutionPlan("block FP8 attention plan received MXFP8 weight"))
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::MxFp4(_)) => {
                Err(Error::InvalidExecutionPlan("block FP8 attention plan received MXFP4 weight"))
            },
            (Self::Fp8Int4, DecodeAttentionOutputWeight::Fp8Int4 { quantized, .. }) => {
                quantized.execute(stream, input, output)
            },
            (Self::Fp8Int4, _) => Err(Error::InvalidExecutionPlan(
                "FP8 plus INT4 attention plan lacks quantized weight",
            )),
            (Self::BlockFp8, DecodeAttentionOutputWeight::Fp8Int4 { .. }) => Err(
                Error::InvalidExecutionPlan("block FP8 attention plan received residual weight"),
            ),
            (_, DecodeAttentionOutputWeight::NvFp4(_)) => {
                Err(Error::InvalidExecutionPlan("NVFP4 attention output execution is not prepared"))
            },
            (_, DecodeAttentionOutputWeight::PackedInteger(_)) => Err(Error::InvalidExecutionPlan(
                "packed integer attention output execution is not prepared",
            )),
            (_, DecodeAttentionOutputWeight::Affine(_)) => Err(Error::InvalidExecutionPlan(
                "affine attention output execution is not prepared",
            )),
        }
    }
}
