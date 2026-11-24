use super::{Bf16Projection, CudaBackend, DecodeAttentionConfig, ProjectionFormat};
use crate::{
    AffineQuantizedWeight, BlockFp8LinearWeight, CudaTensor, DenseExecution, DensePlanRequest,
    DenseRole, DirectFp8Bf16Linear, DirectFp8CheckpointWeight, Error, ExecutionPhase,
    Fp8ResidualLinearWeight, GgmlBf16Linear, GgmlCheckpointWeight, MxFp4Bf16Linear,
    MxFp4CheckpointWeight, MxFp8Bf16Linear, MxFp8CheckpointWeight, NvFp4Bf16Linear,
    NvFp4LinearWeight, PackedIntegerBf16Linear, PackedIntegerWeight, Result,
};

mod execute;

#[derive(Clone, Copy)]
pub enum DecodeAttentionOutputWeight<'a> {
    Affine(&'a AffineQuantizedWeight),
    Bf16(&'a CudaTensor),
    DirectFp8(&'a DirectFp8CheckpointWeight),
    MxFp4(&'a MxFp4CheckpointWeight),
    Ggml(&'a GgmlCheckpointWeight),
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
            | Self::Ggml(_)
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
    Ggml(GgmlBf16Linear),
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
        let request =
            request(DenseRole::AttentionOutput, tokens, input_features, config.hidden_size);
        Self::prepare(backend, config.projection_format, request, weight)
    }

    /// The query projection of a layer that reads another layer's keys and
    /// values: one projection of the normalized stream, in the layer's
    /// format.
    pub(super) fn new_query(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        tokens: usize,
        weight: Option<DecodeAttentionOutputWeight<'_>>,
    ) -> Result<Self> {
        let query = config.query_heads * config.cache.key_head_dim;
        let request = request(DenseRole::AttentionQkv, tokens, config.hidden_size, query);
        if config.projection_format == ProjectionFormat::Bf16 {
            return Ok(Self::Bf16(backend.prepare_bf16_projection(request)?));
        }
        Self::prepare(backend, config.projection_format, request, weight)
    }

    fn prepare(
        backend: &CudaBackend,
        format: ProjectionFormat,
        request: DensePlanRequest,
        weight: Option<DecodeAttentionOutputWeight<'_>>,
    ) -> Result<Self> {
        let (tokens, role) = (request.tokens, request.role);
        let (input_features, output_features) = (request.input_features, request.output_features);
        match format {
            ProjectionFormat::Affine => {
                let Some(DecodeAttentionOutputWeight::Affine(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "affine attention requires affine output weight",
                    ));
                };
                let affine = weight.infer_config(1, input_features, output_features)?;
                Ok(Self::Affine(crate::backend::linear::AffineProjection::new(
                    backend,
                    tokens,
                    input_features,
                    output_features,
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
                    backend, tokens, input_features, output_features, weight,
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
            ProjectionFormat::Ggml => {
                let Some(DecodeAttentionOutputWeight::Ggml(weight)) = weight else {
                    return Err(Error::InvalidExecutionPlan(
                        "GGML attention requires prepared output weight",
                    ));
                };
                Ok(Self::Ggml(weight.prepare(backend, tokens, role)?))
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
                    DenseExecution::Matrix
                    | DenseExecution::Vector
                    | DenseExecution::RowGroupVector
                    | DenseExecution::CublasLt => {
                        Self::Bf16(backend.prepare_bf16_projection(request)?)
                    },
                },
            ),
        }
    }
}

const fn request(
    role: DenseRole,
    tokens: usize,
    input_features: usize,
    output_features: usize,
) -> DensePlanRequest {
    DensePlanRequest {
        phase: if tokens == 1 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        },
        role,
        tokens,
        input_features,
        output_features,
    }
}
