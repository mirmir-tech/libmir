use super::DenseDownSource;
use crate::{
    AffineQuantizedWeight, BlockFp8LinearWeight, CudaBackend, CudaTensor, DenseDownWeight,
    DenseExecution, DensePlanRequest, DenseRole, DirectFp8CheckpointWeight, Error, ExecutionPhase,
    Fp8ResidualLinearWeight, MxFp4CheckpointWeight, MxFp8CheckpointWeight, NvFp4LinearWeight,
    PackedIntegerWeight, Result,
};

#[derive(Clone)]
pub(super) enum DenseDownOwned {
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

impl DenseDownOwned {
    pub(super) fn new(backend: &CudaBackend, source: DenseDownSource<'_>) -> Result<Self> {
        let DenseDownSource::Bf16(weight) = source else {
            return Ok(match source {
                DenseDownSource::Affine(weight) => Self::Affine(weight.clone()),
                DenseDownSource::PackedInteger(weight) => Self::PackedInteger(weight.clone()),
                DenseDownSource::DirectFp8(weight) => Self::DirectFp8(weight.clone()),
                DenseDownSource::MxFp4(weight) => Self::MxFp4(weight.clone()),
                DenseDownSource::MxFp8(weight) => Self::MxFp8(weight.clone()),
                DenseDownSource::NvFp4(weight) => Self::NvFp4(weight.clone()),
                DenseDownSource::Bf16(_) => unreachable!(),
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
            role: DenseRole::DenseDown,
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

    pub(super) fn borrow(&self) -> DenseDownWeight<'_> {
        match self {
            Self::Affine(weight) => DenseDownWeight::Affine(weight),
            Self::Bf16(weight) => DenseDownWeight::Bf16(weight),
            Self::DirectFp8(weight) => DenseDownWeight::DirectFp8(weight),
            Self::MxFp4(weight) => DenseDownWeight::MxFp4(weight),
            Self::MxFp8(weight) => DenseDownWeight::MxFp8(weight),
            Self::PackedInteger(weight) => DenseDownWeight::PackedInteger(weight),
            Self::NvFp4(weight) => DenseDownWeight::NvFp4(weight),
            Self::BlockFp8 { exact, quantized } => DenseDownWeight::BlockFp8 { exact, quantized },
            Self::Fp8Int4 { exact, quantized } => DenseDownWeight::Fp8Int4 { exact, quantized },
        }
    }
}
