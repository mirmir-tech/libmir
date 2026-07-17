use mircuda::{
    DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan, DenseVectorSpec, DeviceBuffer, Stream, bf16,
};

use super::CudaBackend;
use crate::{
    CudaTensor, DenseExecution, DensePlanRequest, DenseRole, Error, ExecutionPhase, Result,
};

/// Device-resident concatenation of two equally shaped BF16 linear weights.
#[derive(Clone, Debug)]
pub struct Bf16LinearPairWeights {
    packed: DeviceBuffer<bf16>,
    input_features: usize,
    output_features: usize,
}

/// One GEMM producing two adjacent projections for every input row.
#[derive(Debug)]
pub struct Bf16LinearPair {
    plan: Bf16LinearPairPlan,
    stream: Stream,
    tokens: usize,
    input_features: usize,
    output_features: usize,
}

#[derive(Debug)]
enum Bf16LinearPairPlan {
    Matrix(DenseMatmulPlan<bf16>),
    Vector(DenseVectorPlan<bf16>),
}

impl CudaBackend {
    /// Concatenates two `[out, in]` BF16 weights once using an asynchronous D2D
    /// copy.
    pub fn pack_bf16_linear_pair(
        &self,
        left: &CudaTensor,
        right: &CudaTensor,
    ) -> Result<Bf16LinearPairWeights> {
        Bf16LinearPairWeights::new(self, left, right)
    }
}

impl Bf16LinearPairWeights {
    fn new(backend: &CudaBackend, left: &CudaTensor, right: &CudaTensor) -> Result<Self> {
        let [output_features, input_features] = left.shape() else {
            return Err(invalid(left, [0, 0]));
        };
        let expected = [*output_features, *input_features];
        if right.shape() != expected {
            return Err(invalid(right, expected));
        }
        let left = bf16_weight(left)?;
        let right = bf16_weight(right)?;
        let elements = output_features
            .checked_mul(*input_features)
            .ok_or(Error::InvalidDecoderKernel("packed BF16 weight size overflow"))?;
        let packed_elements = elements
            .checked_mul(2)
            .ok_or(Error::InvalidDecoderKernel("packed BF16 weight size overflow"))?;
        let mut packed =
            backend.inner.pool.allocate::<bf16>(&backend.inner.stream, packed_elements)?;
        backend.inner.stream.copy_device_range(left, 0..elements, &mut packed, 0)?;
        backend
            .inner
            .stream
            .copy_device_range(right, 0..elements, &mut packed, elements)?;
        Ok(Self {
            packed,
            input_features: *input_features,
            output_features: *output_features,
        })
    }
}

impl Bf16LinearPair {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        phase: ExecutionPhase,
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        let packed_outputs = output_features
            .checked_mul(2)
            .ok_or(Error::InvalidDecoderKernel("paired BF16 output size overflow"))?;
        let request = DensePlanRequest {
            phase,
            role: DenseRole::DenseGateUp,
            tokens,
            input_features,
            output_features: packed_outputs,
        };
        let plan = match backend.execution_planner().plan_dense(request)?.execution() {
            DenseExecution::Matrix => Bf16LinearPairPlan::Matrix(DenseMatmulPlan::new(
                &backend.inner.context,
                &backend.inner.stream,
                DenseMatmulSpec::new(tokens, packed_outputs, input_features)?,
            )?),
            DenseExecution::Vector => Bf16LinearPairPlan::Vector(DenseVectorPlan::new(
                &backend.inner.context,
                &backend.inner.stream,
                DenseVectorSpec::new(packed_outputs, input_features)?,
            )?),
            DenseExecution::BlockFp8Vector => {
                return Err(Error::InvalidExecutionPlan(
                    "paired BF16 projection cannot use block FP8 weights",
                ));
            },
            DenseExecution::Fp8Int4Vector => {
                return Err(Error::InvalidExecutionPlan(
                    "paired BF16 projection cannot use FP8 plus INT4 weights",
                ));
            },
        };
        Ok(Self {
            plan,
            stream: backend.inner.stream.clone(),
            tokens,
            input_features,
            output_features,
        })
    }

    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: &Bf16LinearPairWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if weights.input_features != self.input_features
            || weights.output_features != self.output_features
        {
            return Err(Error::InvalidDecoderKernel("paired BF16 weights differ from plan"));
        }
        let expected = self
            .tokens
            .checked_mul(self.output_features)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or(Error::InvalidDecoderKernel("paired BF16 output size overflow"))?;
        if output.len() != expected {
            return Err(Error::InvalidDecoderKernel("paired BF16 output differs from plan"));
        }
        match &mut self.plan {
            Bf16LinearPairPlan::Matrix(plan) => {
                Ok(plan.execute(&self.stream, input, &weights.packed, output, 1.0, 0.0)?)
            },
            Bf16LinearPairPlan::Vector(plan) => {
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
