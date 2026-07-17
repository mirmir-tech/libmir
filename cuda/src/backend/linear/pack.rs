use mircuda::{
    DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan, DenseVectorSpec, DeviceBuffer, Stream, bf16,
};

use super::CudaBackend;
use crate::{
    CudaTensor, DenseExecution, DensePlanRequest, DenseRole, Error, ExecutionPhase, Result,
};

/// Device-resident row concatenation of differently sized BF16 projections.
#[derive(Clone, Debug)]
pub struct Bf16LinearPackWeights<const N: usize> {
    packed: DeviceBuffer<bf16>,
    input_features: usize,
    output_features: [usize; N],
}

/// One GEMM producing `N` adjacent projections for every input row.
#[derive(Debug)]
pub struct Bf16LinearPack<const N: usize> {
    plan: Bf16LinearPackPlan,
    stream: Stream,
    tokens: usize,
    input_features: usize,
    output_features: [usize; N],
    packed_outputs: usize,
}

#[derive(Debug)]
enum Bf16LinearPackPlan {
    Matrix(DenseMatmulPlan<bf16>),
    Vector(DenseVectorPlan<bf16>),
}

impl CudaBackend {
    /// Packs `[out, in]` BF16 weights once using asynchronous D2D copies.
    pub fn pack_bf16_linears<const N: usize>(
        &self,
        weights: [&CudaTensor; N],
    ) -> Result<Bf16LinearPackWeights<N>> {
        Bf16LinearPackWeights::new(self, weights)
    }
}

impl<const N: usize> Bf16LinearPackWeights<N> {
    fn new(backend: &CudaBackend, weights: [&CudaTensor; N]) -> Result<Self> {
        let first = weights
            .first()
            .ok_or(Error::InvalidDecoderKernel("BF16 linear pack cannot be empty"))?;
        let [_, input_features] = first.shape() else {
            return Err(invalid(first, [0, 0]));
        };
        let mut output_features = [0; N];
        let mut packed_outputs = 0_usize;
        for (index, weight) in weights.iter().enumerate() {
            let [output, input] = weight.shape() else {
                return Err(invalid(weight, [0, *input_features]));
            };
            if input != input_features {
                return Err(invalid(weight, [*output, *input_features]));
            }
            bf16_weight(weight)?;
            output_features[index] = *output;
            packed_outputs = packed_outputs
                .checked_add(*output)
                .ok_or(Error::InvalidDecoderKernel("packed BF16 output size overflow"))?;
        }
        let packed_elements = packed_outputs
            .checked_mul(*input_features)
            .ok_or(Error::InvalidDecoderKernel("packed BF16 weight size overflow"))?;
        let mut packed =
            backend.inner.pool.allocate::<bf16>(&backend.inner.stream, packed_elements)?;
        let mut offset = 0_usize;
        for weight in weights {
            let source = bf16_weight(weight)?;
            backend
                .inner
                .stream
                .copy_device_range(source, 0..source.len(), &mut packed, offset)?;
            offset += source.len();
        }
        Ok(Self {
            packed,
            input_features: *input_features,
            output_features,
        })
    }

    #[must_use]
    pub const fn output_features(&self) -> [usize; N] {
        self.output_features
    }
}

impl<const N: usize> Bf16LinearPack<N> {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        phase: ExecutionPhase,
        tokens: usize,
        input_features: usize,
        output_features: [usize; N],
    ) -> Result<Self> {
        let packed_outputs = output_features.iter().try_fold(0_usize, |total, width| {
            total
                .checked_add(*width)
                .ok_or(Error::InvalidDecoderKernel("packed BF16 output size overflow"))
        })?;
        if packed_outputs == 0 {
            return Err(Error::InvalidDecoderKernel("BF16 linear pack cannot be empty"));
        }
        let request = DensePlanRequest {
            phase,
            role: DenseRole::AttentionQkv,
            tokens,
            input_features,
            output_features: packed_outputs,
        };
        let plan = match backend.execution_planner().plan_dense(request)?.execution() {
            DenseExecution::Matrix => Bf16LinearPackPlan::Matrix(DenseMatmulPlan::new(
                &backend.inner.context,
                &backend.inner.stream,
                DenseMatmulSpec::new(tokens, packed_outputs, input_features)?,
            )?),
            DenseExecution::Vector => Bf16LinearPackPlan::Vector(DenseVectorPlan::new(
                &backend.inner.context,
                &backend.inner.stream,
                DenseVectorSpec::new(packed_outputs, input_features)?,
            )?),
            DenseExecution::BlockFp8Vector => {
                return Err(Error::InvalidExecutionPlan(
                    "packed BF16 projection cannot use block FP8 weights",
                ));
            },
            DenseExecution::Fp8Int4Vector => {
                return Err(Error::InvalidExecutionPlan(
                    "packed BF16 projection cannot use FP8 plus INT4 weights",
                ));
            },
        };
        Ok(Self {
            plan,
            stream: backend.inner.stream.clone(),
            tokens,
            input_features,
            output_features,
            packed_outputs,
        })
    }

    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: &Bf16LinearPackWeights<N>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if weights.input_features != self.input_features
            || weights.output_features != self.output_features
        {
            return Err(Error::InvalidDecoderKernel("packed BF16 weights differ from plan"));
        }
        let expected = self
            .tokens
            .checked_mul(self.packed_outputs)
            .ok_or(Error::InvalidDecoderKernel("packed BF16 output size overflow"))?;
        if output.len() != expected {
            return Err(Error::InvalidDecoderKernel("packed BF16 output differs from plan"));
        }
        match &mut self.plan {
            Bf16LinearPackPlan::Matrix(plan) => {
                Ok(plan.execute(&self.stream, input, &weights.packed, output, 1.0, 0.0)?)
            },
            Bf16LinearPackPlan::Vector(plan) => {
                Ok(plan.execute(&self.stream, input, &weights.packed, output, 1.0, 0.0)?)
            },
        }
    }
}

fn bf16_weight(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}

fn invalid(tensor: &CudaTensor, expected: [usize; 2]) -> Error {
    Error::InvalidLinearWeight {
        name: tensor.name().into(),
        expected,
        actual: tensor.shape().to_vec(),
    }
}
