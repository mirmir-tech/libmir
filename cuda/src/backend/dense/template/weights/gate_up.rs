use super::DenseGateUpSource;
use crate::{
    AffineQuantizedWeight, Bf16LinearPairWeights, BlockFp8LinearWeight, CudaBackend,
    DenseExecution, DenseGateUpWeights, DensePlanRequest, DenseRole, DirectFp8CheckpointWeight,
    ExecutionPhase, Fp8ResidualLinearWeight, MxFp4CheckpointWeight, MxFp8CheckpointWeight,
    NvFp4LinearWeight, PackedIntegerWeight, Result,
};

#[derive(Clone)]
pub(super) enum DenseGateUpOwned {
    Affine {
        gate: AffineQuantizedWeight,
        up: AffineQuantizedWeight,
    },
    Bf16(Bf16LinearPairWeights),
    DirectFp8 {
        gate: DirectFp8CheckpointWeight,
        up: DirectFp8CheckpointWeight,
    },
    MxFp4 {
        gate: MxFp4CheckpointWeight,
        up: MxFp4CheckpointWeight,
    },
    MxFp8 {
        gate: MxFp8CheckpointWeight,
        up: MxFp8CheckpointWeight,
    },
    PackedInteger {
        gate: PackedIntegerWeight,
        up: PackedIntegerWeight,
    },
    NvFp4 {
        gate: NvFp4LinearWeight,
        up: NvFp4LinearWeight,
    },
    BlockFp8 {
        exact: Bf16LinearPairWeights,
        quantized: BlockFp8LinearWeight,
    },
    Fp8Int4 {
        exact: Bf16LinearPairWeights,
        quantized: Fp8ResidualLinearWeight,
    },
}

impl DenseGateUpOwned {
    pub(super) fn new(backend: &CudaBackend, source: DenseGateUpSource<'_>) -> Result<Self> {
        let DenseGateUpSource::Bf16(weights) = source else {
            return Ok(match source {
                DenseGateUpSource::Affine { gate, up } => {
                    Self::Affine { gate: gate.clone(), up: up.clone() }
                },
                DenseGateUpSource::PackedInteger { gate, up } => {
                    Self::PackedInteger { gate: gate.clone(), up: up.clone() }
                },
                DenseGateUpSource::DirectFp8 { gate, up } => {
                    Self::DirectFp8 { gate: gate.clone(), up: up.clone() }
                },
                DenseGateUpSource::MxFp4 { gate, up } => {
                    Self::MxFp4 { gate: gate.clone(), up: up.clone() }
                },
                DenseGateUpSource::MxFp8 { gate, up } => {
                    Self::MxFp8 { gate: gate.clone(), up: up.clone() }
                },
                DenseGateUpSource::NvFp4 { gate, up } => {
                    Self::NvFp4 { gate: gate.clone(), up: up.clone() }
                },
                DenseGateUpSource::Bf16(_) => unreachable!(),
            });
        };
        let request = DensePlanRequest {
            phase: ExecutionPhase::Decode,
            role: DenseRole::DenseGateUp,
            tokens: 1,
            input_features: weights.input_features(),
            output_features: weights.packed_output_features()?,
        };
        Ok(
            match backend
                .execution_planner()
                .plan_dense_with_prepared_weights(request)?
                .execution()
            {
                DenseExecution::BlockFp8Vector => Self::BlockFp8 {
                    exact: weights.clone(),
                    quantized: backend.prepare_block_fp8_linear_pair_weight(weights)?,
                },
                DenseExecution::Fp8Int4Vector => Self::Fp8Int4 {
                    exact: weights.clone(),
                    quantized: backend.prepare_fp8_residual_linear_pair_weight(weights)?,
                },
                DenseExecution::Matrix | DenseExecution::Vector | DenseExecution::CublasLt => {
                    Self::Bf16(weights.clone())
                },
            },
        )
    }

    pub(super) fn borrow(&self) -> DenseGateUpWeights<'_> {
        match self {
            Self::Affine { gate, up } => DenseGateUpWeights::Affine { gate, up },
            Self::Bf16(weights) => DenseGateUpWeights::Bf16(weights),
            Self::DirectFp8 { gate, up } => DenseGateUpWeights::DirectFp8 { gate, up },
            Self::MxFp4 { gate, up } => DenseGateUpWeights::MxFp4 { gate, up },
            Self::MxFp8 { gate, up } => DenseGateUpWeights::MxFp8 { gate, up },
            Self::PackedInteger { gate, up } => DenseGateUpWeights::PackedInteger { gate, up },
            Self::NvFp4 { gate, up } => DenseGateUpWeights::NvFp4 { gate, up },
            Self::BlockFp8 { exact, quantized } => {
                DenseGateUpWeights::BlockFp8 { exact, quantized }
            },
            Self::Fp8Int4 { exact, quantized } => DenseGateUpWeights::Fp8Int4 { exact, quantized },
        }
    }
}
