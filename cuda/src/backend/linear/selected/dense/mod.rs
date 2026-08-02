use mircuda::{DeviceBuffer, Stream, bf16};

use super::super::CudaBackend;
use crate::{
    Result,
    kernels::{
        DenseGatedActivation, SelectedDenseDispatch, SelectedDenseGateLaunch, SelectedDenseMoe,
        SelectedDenseReduceLaunch,
    },
};

mod canonical;
mod tensor_core;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod weights;

use tensor_core::TensorCoreScratch;
pub use weights::DenseExpertWeights;

#[derive(Debug)]
pub(in crate::backend) struct SelectedDenseMoeBf16 {
    operation: SelectedDenseMoe,
    down_partial: DeviceBuffer<f32>,
    dispatch: Option<ExpertMajorScratch>,
    tensor_core: Option<TensorCoreScratch>,
    stream: Stream,
}

#[derive(Debug)]
struct ExpertMajorScratch {
    counts: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    cursors: DeviceBuffer<u32>,
    assignments: DeviceBuffer<u32>,
    experts: DeviceBuffer<u32>,
}

impl SelectedDenseMoeBf16 {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected_count: usize,
        weights: &DenseExpertWeights,
        activation: DenseGatedActivation,
    ) -> Result<Self> {
        let spec = weights.spec(tokens, selected_count, activation)?;
        let operation = SelectedDenseMoe::compile(&backend.inner.compiler, spec)?;
        let expert_major = operation.prefers_expert_major();
        let assignments = tokens
            .checked_mul(selected_count)
            .ok_or(crate::Error::InvalidDecoderKernel("dense expert assignment size overflow"))?;
        let partial_elements = if tokens == 1 || expert_major {
            assignments.checked_mul(spec.input_features).ok_or(
                crate::Error::InvalidDecoderKernel(
                    "dense selected-expert down scratch size overflow",
                ),
            )?
        } else {
            1
        };
        let tensor_core =
            expert_major.then(|| TensorCoreScratch::new(backend, &operation)).transpose()?;
        Ok(Self {
            operation,
            down_partial: backend.inner.pool.allocate(&backend.inner.stream, partial_elements)?,
            dispatch: tensor_core
                .is_some()
                .then(|| ExpertMajorScratch::new(backend, spec.expert_count, assignments))
                .transpose()?,
            tensor_core,
            stream: backend.inner.stream.clone(),
        })
    }

    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &DenseExpertWeights,
        intermediate: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let (gate, up) = weights.gate_up();
        let gate_weight = gate.weight.as_bf16().ok_or_else(|| dtype(&gate.weight))?;
        let up_weight = up.weight.as_bf16().ok_or_else(|| dtype(&up.weight))?;
        let fallback = input;
        let gate_bias = gate.bias.as_ref().and_then(crate::CudaTensor::as_bf16).unwrap_or(fallback);
        let up_bias = up.bias.as_ref().and_then(crate::CudaTensor::as_bf16).unwrap_or(fallback);
        let mut gate = SelectedDenseGateLaunch {
            input,
            selected,
            gate_weight,
            gate_bias,
            up_weight,
            up_bias,
            output: intermediate,
        };
        let mut dispatch = self.dispatch.as_mut().map(ExpertMajorScratch::borrow);
        if let Some(dispatch) = dispatch.as_mut() {
            self.operation.prepare_expert_major(&self.stream, selected, dispatch)?;
            let down = &weights.down;
            let down_weight = down.weight.as_bf16().ok_or_else(|| dtype(&down.weight))?;
            let down_bias =
                down.bias.as_ref().and_then(crate::CudaTensor::as_bf16).unwrap_or(fallback);
            return self
                .tensor_core
                .as_mut()
                .ok_or(crate::Error::InvalidExecutionPlan(
                    "dense grouped tensor-core execution was not prepared",
                ))?
                .execute(
                    &self.operation,
                    &self.stream,
                    input,
                    selected,
                    routing,
                    gate_weight,
                    gate_bias,
                    down_weight,
                    down_bias,
                    intermediate,
                    &mut self.down_partial,
                    output,
                    dispatch,
                );
        }
        self.operation.gated(&self.stream, &mut gate)?;
        self.reduce(selected, routing, weights, intermediate, output, fallback)
    }

    fn reduce(
        &mut self,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &DenseExpertWeights,
        intermediate: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        fallback: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        let down = &weights.down;
        let down_weight = down.weight.as_bf16().ok_or_else(|| dtype(&down.weight))?;
        let down_bias = down.bias.as_ref().and_then(crate::CudaTensor::as_bf16).unwrap_or(fallback);
        self.operation.reduce(
            &self.stream,
            &mut SelectedDenseReduceLaunch {
                input: intermediate,
                selected,
                routing,
                weight: down_weight,
                bias: down_bias,
                partial: &mut self.down_partial,
                output,
            },
        )
    }
}

impl ExpertMajorScratch {
    fn new(backend: &CudaBackend, experts: usize, assignments: usize) -> Result<Self> {
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            counts: allocate(experts)?,
            offsets: allocate(experts)?,
            cursors: allocate(experts)?,
            assignments: allocate(assignments)?,
            experts: allocate(assignments)?,
        })
    }

    fn borrow(&mut self) -> SelectedDenseDispatch<'_> {
        SelectedDenseDispatch {
            counts: &mut self.counts,
            offsets: &mut self.offsets,
            cursors: &mut self.cursors,
            assignments: &mut self.assignments,
            experts: &mut self.experts,
        }
    }
}

fn dtype(tensor: &crate::CudaTensor) -> crate::Error {
    crate::Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    }
}
