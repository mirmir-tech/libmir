use mircuda::{DeviceBuffer, Stream, bf16};

use super::{Bf16Projection, CudaBackend, DecodeAttentionConfig, ProjectionFormat};
use crate::{
    BlockFp8LinearWeight, CompressedInt8Bf16Linear, CompressedInt8Weight, CudaTensor,
    DenseExecution, DensePlanRequest, DenseRole, Error, ExecutionPhase, Fp8ResidualLinearWeight,
    NvFp4Bf16Linear, NvFp4LinearWeight, Result,
};

#[derive(Clone, Copy)]
pub enum DecodeAttentionOutputWeight<'a> {
    Bf16(&'a CudaTensor),
    Int8(&'a CompressedInt8Weight),
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
            Self::Int8(_) | Self::NvFp4(_) => None,
        }
    }
}

#[derive(Debug)]
pub(super) enum AttentionOutputProjection {
    Bf16(Bf16Projection),
    Int8(CompressedInt8Bf16Linear),
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
            ProjectionFormat::Int8 => {
                let Some(DecodeAttentionOutputWeight::Int8(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "INT8 attention requires prepared output weight",
                    ));
                };
                Ok(Self::Int8(CompressedInt8Bf16Linear::new(
                    backend,
                    tokens,
                    input_features,
                    config.hidden_size,
                    weight.clone(),
                )?))
            },
            ProjectionFormat::NvFp4 => {
                let Some(DecodeAttentionOutputWeight::NvFp4(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "NVFP4 attention requires prepared output weight",
                    ));
                };
                Ok(Self::NvFp4(NvFp4Bf16Linear::from_weight(backend, tokens, weight.clone())?))
            },
            ProjectionFormat::Bf16 => {
                Ok(match backend.execution_planner().plan_dense(request)?.execution() {
                    DenseExecution::BlockFp8Vector => Self::BlockFp8,
                    DenseExecution::Fp8Int4Vector => Self::Fp8Int4,
                    DenseExecution::Matrix | DenseExecution::Vector => {
                        Self::Bf16(backend.prepare_bf16_projection(request)?)
                    },
                })
            },
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
            (Self::Int8(operation), DecodeAttentionOutputWeight::Int8(_)) => {
                operation.execute(input, output)
            },
            (Self::Int8(_), _) => {
                Err(Error::InvalidExecutionPlan("INT8 attention output received other weight"))
            },
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
            (Self::BlockFp8, DecodeAttentionOutputWeight::BlockFp8 { quantized, .. }) => {
                quantized.execute(stream, input, output)
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::Bf16(_)) => {
                Err(Error::InvalidExecutionPlan("block FP8 attention plan lacks quantized weight"))
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
            (_, DecodeAttentionOutputWeight::Int8(_)) => {
                Err(Error::InvalidExecutionPlan("INT8 attention output execution is not prepared"))
            },
        }
    }
}
