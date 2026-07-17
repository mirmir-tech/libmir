use mircuda::{DeviceBuffer, bf16};

use crate::{
    Bf16LinearPackWeights, BlockFp8LinearWeight, CudaBackend, CudaTensor,
    DecodeAttentionOutputWeight, DecodeAttentionWeights, DecodeQkvWeights, DenseExecution,
    DensePlanRequest, DenseRole, Error, ExecutionPhase, Fp8ResidualLinearWeight, NvFp4LinearWeight,
    Result,
};

#[derive(Clone)]
enum CapturedOutputWeight {
    Bf16(CudaTensor),
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
    Bf16(Bf16LinearPackWeights<3>),
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
        let output = match backend.execution_planner().plan_dense(request)?.execution() {
            DenseExecution::BlockFp8Vector => CapturedOutputWeight::BlockFp8 {
                exact: output.clone(),
                quantized: backend.prepare_block_fp8_linear_weight(output)?,
            },
            DenseExecution::Fp8Int4Vector => CapturedOutputWeight::Fp8Int4 {
                exact: output.clone(),
                quantized: backend.prepare_fp8_residual_linear_weight(output)?,
            },
            DenseExecution::Matrix | DenseExecution::Vector => {
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
                CapturedOutputWeight::Bf16(weight) => DecodeAttentionOutputWeight::Bf16(weight),
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
            DecodeQkvWeights::Bf16(weights) => Self::Bf16(weights.clone()),
            DecodeQkvWeights::NvFp4(weights) => Self::NvFp4(weights.map(Clone::clone)),
        }
    }
}

impl CapturedQkvWeights {
    fn borrow(&self) -> DecodeQkvWeights<'_> {
        match self {
            Self::Bf16(weights) => DecodeQkvWeights::Bf16(weights),
            Self::NvFp4(weights) => {
                DecodeQkvWeights::NvFp4([&weights[0], &weights[1], &weights[2]])
            },
        }
    }
}

impl From<DecodeAttentionOutputWeight<'_>> for CapturedOutputWeight {
    fn from(weight: DecodeAttentionOutputWeight<'_>) -> Self {
        match weight {
            DecodeAttentionOutputWeight::Bf16(weight) => Self::Bf16(weight.clone()),
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
