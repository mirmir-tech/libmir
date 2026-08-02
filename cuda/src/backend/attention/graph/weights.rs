use mircuda::{DeviceBuffer, bf16};

use crate::{
    AffineQuantizedWeight, Bf16LinearPackWeights, BlockFp8LinearWeight, CudaBackend, CudaTensor,
    DecodeAttentionOutputWeight, DecodeAttentionWeights, DecodeQkvWeights, DenseExecution,
    DensePlanRequest, DenseRole, DirectFp8CheckpointWeight, Error, ExecutionPhase,
    Fp8ResidualLinearWeight, MxFp4CheckpointWeight, MxFp8CheckpointWeight, NvFp4LinearWeight,
    PackedIntegerWeight, Result,
};

#[derive(Clone)]
enum CapturedOutputWeight {
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

#[derive(Clone)]
enum CapturedQkvWeights {
    Affine(Box<[AffineQuantizedWeight; 3]>),
    Bf16(Bf16LinearPackWeights<3>),
    DirectFp8(Box<[DirectFp8CheckpointWeight; 3]>),
    MxFp4(Box<[MxFp4CheckpointWeight; 3]>),
    MxFp8(Box<[MxFp8CheckpointWeight; 3]>),
    PackedInteger(Box<[PackedIntegerWeight; 3]>),
    NvFp4([NvFp4LinearWeight; 3]),
}

#[derive(Clone)]
pub(in crate::backend) struct CapturedAttentionWeights {
    input_norm: CudaTensor,
    qkv: CapturedQkvWeights,
    query_norm: DeviceBuffer<bf16>,
    key_norm: DeviceBuffer<bf16>,
    output: CapturedOutputWeight,
}

impl CapturedAttentionWeights {
    pub(in crate::backend) fn prepare(
        backend: &CudaBackend,
        weights: DecodeAttentionWeights<'_>,
    ) -> Result<Self> {
        let output = weights.output.bf16().ok_or(Error::InvalidExecutionPlan(
            "optimized BF16 graph preparation received NVFP4 output weight",
        ))?;
        let [output_features, input_features] = output.shape() else {
            return Err(Error::InvalidLinearWeight {
                name: output.name().into(),
                expected: [0, 0],
                actual: output.shape().to_vec(),
            });
        };
        let request = DensePlanRequest {
            phase: ExecutionPhase::Decode,
            role: DenseRole::AttentionOutput,
            tokens: 1,
            input_features: *input_features,
            output_features: *output_features,
        };
        let output = match backend
            .execution_planner()
            .plan_dense_with_prepared_weights(request)?
            .execution()
        {
            DenseExecution::BlockFp8Vector => CapturedOutputWeight::BlockFp8 {
                exact: output.clone(),
                quantized: backend.prepare_block_fp8_linear_weight(output)?,
            },
            DenseExecution::Fp8Int4Vector => CapturedOutputWeight::Fp8Int4 {
                exact: output.clone(),
                quantized: backend.prepare_fp8_residual_linear_weight(output)?,
            },
            DenseExecution::Matrix | DenseExecution::Vector | DenseExecution::CublasLt => {
                CapturedOutputWeight::Bf16(output.clone())
            },
        };
        Ok(Self {
            input_norm: weights.input_norm.clone(),
            qkv: CapturedQkvWeights::from(weights.qkv),
            query_norm: weights.query_norm.clone(),
            key_norm: weights.key_norm.clone(),
            output,
        })
    }

    pub(in crate::backend) fn borrow(&self) -> DecodeAttentionWeights<'_> {
        DecodeAttentionWeights {
            input_norm: &self.input_norm,
            qkv: self.qkv.borrow(),
            query_norm: &self.query_norm,
            key_norm: &self.key_norm,
            output: match &self.output {
                CapturedOutputWeight::Affine(weight) => DecodeAttentionOutputWeight::Affine(weight),
                CapturedOutputWeight::Bf16(weight) => DecodeAttentionOutputWeight::Bf16(weight),
                CapturedOutputWeight::DirectFp8(weight) => {
                    DecodeAttentionOutputWeight::DirectFp8(weight)
                },
                CapturedOutputWeight::MxFp4(weight) => DecodeAttentionOutputWeight::MxFp4(weight),
                CapturedOutputWeight::MxFp8(weight) => DecodeAttentionOutputWeight::MxFp8(weight),
                CapturedOutputWeight::PackedInteger(weight) => {
                    DecodeAttentionOutputWeight::PackedInteger(weight)
                },
                CapturedOutputWeight::NvFp4(weight) => DecodeAttentionOutputWeight::NvFp4(weight),
                CapturedOutputWeight::BlockFp8 { exact, quantized } => {
                    DecodeAttentionOutputWeight::BlockFp8 { exact, quantized }
                },
                CapturedOutputWeight::Fp8Int4 { exact, quantized } => {
                    DecodeAttentionOutputWeight::Fp8Int4 { exact, quantized }
                },
            },
        }
    }
}

impl From<DecodeAttentionWeights<'_>> for CapturedAttentionWeights {
    fn from(weights: DecodeAttentionWeights<'_>) -> Self {
        Self {
            input_norm: weights.input_norm.clone(),
            qkv: CapturedQkvWeights::from(weights.qkv),
            query_norm: weights.query_norm.clone(),
            key_norm: weights.key_norm.clone(),
            output: CapturedOutputWeight::from(weights.output),
        }
    }
}

impl From<DecodeQkvWeights<'_>> for CapturedQkvWeights {
    fn from(weights: DecodeQkvWeights<'_>) -> Self {
        match weights {
            DecodeQkvWeights::Affine(weights) => Self::Affine(Box::new(weights.map(Clone::clone))),
            DecodeQkvWeights::Bf16(weights) => Self::Bf16(weights.clone()),
            DecodeQkvWeights::DirectFp8(weights) => {
                Self::DirectFp8(Box::new(weights.map(Clone::clone)))
            },
            DecodeQkvWeights::MxFp4(weights) => Self::MxFp4(Box::new(weights.map(Clone::clone))),
            DecodeQkvWeights::MxFp8(weights) => Self::MxFp8(Box::new(weights.map(Clone::clone))),
            DecodeQkvWeights::PackedInteger(weights) => {
                Self::PackedInteger(Box::new(weights.map(Clone::clone)))
            },
            DecodeQkvWeights::NvFp4(weights) => Self::NvFp4(weights.map(Clone::clone)),
        }
    }
}

impl CapturedQkvWeights {
    fn borrow(&self) -> DecodeQkvWeights<'_> {
        match self {
            Self::Affine(weights) => {
                DecodeQkvWeights::Affine([&weights[0], &weights[1], &weights[2]])
            },
            Self::Bf16(weights) => DecodeQkvWeights::Bf16(weights),
            Self::DirectFp8(weights) => {
                DecodeQkvWeights::DirectFp8([&weights[0], &weights[1], &weights[2]])
            },
            Self::MxFp4(weights) => {
                DecodeQkvWeights::MxFp4([&weights[0], &weights[1], &weights[2]])
            },
            Self::MxFp8(weights) => {
                DecodeQkvWeights::MxFp8([&weights[0], &weights[1], &weights[2]])
            },
            Self::PackedInteger(weights) => {
                DecodeQkvWeights::PackedInteger([&weights[0], &weights[1], &weights[2]])
            },
            Self::NvFp4(weights) => {
                DecodeQkvWeights::NvFp4([&weights[0], &weights[1], &weights[2]])
            },
        }
    }
}

impl From<DecodeAttentionOutputWeight<'_>> for CapturedOutputWeight {
    fn from(weight: DecodeAttentionOutputWeight<'_>) -> Self {
        match weight {
            DecodeAttentionOutputWeight::Affine(weight) => Self::Affine(weight.clone()),
            DecodeAttentionOutputWeight::Bf16(weight) => Self::Bf16(weight.clone()),
            DecodeAttentionOutputWeight::DirectFp8(weight) => Self::DirectFp8(weight.clone()),
            DecodeAttentionOutputWeight::MxFp4(weight) => Self::MxFp4(weight.clone()),
            DecodeAttentionOutputWeight::MxFp8(weight) => Self::MxFp8(weight.clone()),
            DecodeAttentionOutputWeight::PackedInteger(weight) => {
                Self::PackedInteger(weight.clone())
            },
            DecodeAttentionOutputWeight::NvFp4(weight) => Self::NvFp4(weight.clone()),
            DecodeAttentionOutputWeight::BlockFp8 { exact, quantized } => Self::BlockFp8 {
                exact: exact.clone(),
                quantized: quantized.clone(),
            },
            DecodeAttentionOutputWeight::Fp8Int4 { exact, quantized } => Self::Fp8Int4 {
                exact: exact.clone(),
                quantized: quantized.clone(),
            },
        }
    }
}
