use super::DenseOutputSource;
use crate::{
    AffineQuantizedWeight, BlockFp8LinearWeight, CudaBackend, CudaTensor,
    DecodeAttentionOutputWeight, DenseExecution, DensePlanRequest, DenseRole,
    DirectFp8CheckpointWeight, Error, ExecutionPhase, Fp8ResidualLinearWeight,
    MxFp4CheckpointWeight, MxFp8CheckpointWeight, NvFp4LinearWeight, PackedIntegerWeight, Result,
};

#[derive(Clone)]
pub(super) enum DenseOutputOwned {
    Affine(AffineQuantizedWeight),
    Bf16(CudaTensor),
    DirectFp8(DirectFp8CheckpointWeight),
    MxFp4(MxFp4CheckpointWeight),
    MxFp8(MxFp8CheckpointWeight),
    PackedInteger(PackedIntegerWeight),
    NvFp4(NvFp4LinearWeight),
    BlockFp8 {
        exact: CudaTensor,
        quantized: BlockFp8LinearWeight,
    },
    Fp8Int4 {
        exact: CudaTensor,
        quantized: Fp8ResidualLinearWeight,
    },
}

impl DenseOutputOwned {
    pub(super) fn new(backend: &CudaBackend, source: DenseOutputSource<'_>) -> Result<Self> {
        let DenseOutputSource::Bf16(weight) = source else {
            return Ok(match source {
                DenseOutputSource::Affine(weight) => Self::Affine(weight.clone()),
                DenseOutputSource::PackedInteger(weight) => Self::PackedInteger(weight.clone()),
                DenseOutputSource::DirectFp8(weight) => Self::DirectFp8(weight.clone()),
                DenseOutputSource::MxFp4(weight) => Self::MxFp4(weight.clone()),
                DenseOutputSource::MxFp8(weight) => Self::MxFp8(weight.clone()),
                DenseOutputSource::NvFp4(weight) => Self::NvFp4(weight.clone()),
                DenseOutputSource::Bf16(_) => unreachable!(),
            });
        };
        let [output_features, input_features] = weight.shape() else {
            return Err(Error::InvalidLinearWeight {
                name: weight.name().into(),
                expected: [0, 0],
                actual: weight.shape().to_vec(),
            });
        };
        let request = DensePlanRequest {
            phase: ExecutionPhase::Decode,
            role: DenseRole::AttentionOutput,
            tokens: 1,
            input_features: *input_features,
            output_features: *output_features,
        };
        Ok(
            match backend
                .execution_planner()
                .plan_dense_with_prepared_weights(request)?
                .execution()
            {
                DenseExecution::BlockFp8Vector => Self::BlockFp8 {
                    exact: weight.clone(),
                    quantized: backend.prepare_block_fp8_linear_weight(weight)?,
                },
                DenseExecution::Fp8Int4Vector => Self::Fp8Int4 {
                    exact: weight.clone(),
                    quantized: backend.prepare_fp8_residual_linear_weight(weight)?,
                },
                DenseExecution::Matrix | DenseExecution::Vector | DenseExecution::CublasLt => {
                    Self::Bf16(weight.clone())
                },
            },
        )
    }

    pub(super) fn borrow(&self) -> DecodeAttentionOutputWeight<'_> {
        match self {
            Self::Affine(weight) => DecodeAttentionOutputWeight::Affine(weight),
            Self::Bf16(weight) => DecodeAttentionOutputWeight::Bf16(weight),
            Self::DirectFp8(weight) => DecodeAttentionOutputWeight::DirectFp8(weight),
            Self::MxFp4(weight) => DecodeAttentionOutputWeight::MxFp4(weight),
            Self::MxFp8(weight) => DecodeAttentionOutputWeight::MxFp8(weight),
            Self::PackedInteger(weight) => DecodeAttentionOutputWeight::PackedInteger(weight),
            Self::NvFp4(weight) => DecodeAttentionOutputWeight::NvFp4(weight),
            Self::BlockFp8 { exact, quantized } => {
                DecodeAttentionOutputWeight::BlockFp8 { exact, quantized }
            },
            Self::Fp8Int4 { exact, quantized } => {
                DecodeAttentionOutputWeight::Fp8Int4 { exact, quantized }
            },
        }
    }
}
