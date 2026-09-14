pub(super) mod scratch;

use std::sync::{Arc, Mutex};

use scratch::DenseScratch;

#[cfg(all(test, target_os = "linux"))]
mod tests;
use mircuda::{DeviceBuffer, Stream, bf16};
use models::{layout::DecoderConfig, weights::DenseFeedForwardBindings};

use crate::{
    CudaBackend, CudaTensorSet, DenseRole, Error, GatedActivation, Result,
    backend::linear::{CheckpointProjection, CheckpointProjectionWeight},
    kernels::ElementwiseBf16,
};

#[derive(Clone, Debug)]
pub(in crate::backend) struct DenseFeedForward {
    backend: CudaBackend,
    hidden: usize,
    intermediate: usize,
    activation: GatedActivation,
    gate: CheckpointProjectionWeight,
    up: CheckpointProjectionWeight,
    down: CheckpointProjectionWeight,
}

impl DenseFeedForward {
    pub(in crate::backend) fn load(
        backend: &CudaBackend,
        decoder: &DecoderConfig,
        tensors: &CudaTensorSet,
        bindings: DenseFeedForwardBindings<'_>,
    ) -> Result<Self> {
        let load =
            |binding| CheckpointProjectionWeight::load_binding_prepared(backend, tensors, binding);
        let value = Self {
            backend: backend.clone(),
            hidden: decoder.hidden_size,
            intermediate: decoder.intermediate_size,
            activation: GatedActivation::try_from(decoder)?,
            gate: load(bindings.gate)?,
            up: load(bindings.up)?,
            down: load(bindings.down)?,
        };
        value.gate.affine_format(1, value.hidden, value.intermediate)?;
        value.up.affine_format(1, value.hidden, value.intermediate)?;
        value.down.affine_format(1, value.intermediate, value.hidden)?;
        Ok(value)
    }

    pub(super) fn prepare(&self, tokens: usize) -> Result<DenseExecution> {
        let elements = tokens
            .checked_mul(self.intermediate)
            .ok_or(Error::InvalidDecoderKernel("dense intermediate size overflow"))?;
        let projection = |input, output, role, weight| {
            CheckpointProjection::new(&self.backend, tokens, input, output, role, weight)
        };
        Ok(DenseExecution {
            backend: self.backend.clone(),
            gate: projection(self.hidden, self.intermediate, DenseRole::DenseGateUp, &self.gate)?,
            up: projection(self.hidden, self.intermediate, DenseRole::DenseGateUp, &self.up)?,
            down: projection(self.intermediate, self.hidden, DenseRole::DenseDown, &self.down)?,
            activation: self.activation,
            elementwise: ElementwiseBf16::compile(&self.backend.inner.compiler, elements)?,
            scratch: self.backend.inner.dense_scratch.acquire(&self.backend, elements)?,
        })
    }
}

#[derive(Debug)]
pub(in crate::backend) struct DenseExecution {
    backend: CudaBackend,
    gate: CheckpointProjection,
    up: CheckpointProjection,
    down: CheckpointProjection,
    activation: GatedActivation,
    elementwise: ElementwiseBf16,
    scratch: Arc<Mutex<DenseScratch>>,
}

impl DenseExecution {
    pub(super) fn stream(&self) -> Stream {
        self.backend.inner.stream.clone()
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        // Every user of this scratch enqueues on the same runtime stream. Hold
        // the lock through the whole MLP so another submission cannot interleave.
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("dense scratch lock is poisoned"))?;
        let DenseScratch { gate_output, up_output, activated } = &mut *scratch;
        self.gate.execute(input, gate_output)?;
        self.up.execute(input, up_output)?;
        self.elementwise.gated(
            &self.backend.inner.stream,
            gate_output,
            up_output,
            activated,
            self.activation.into(),
        )?;
        self.down.execute(activated, output)?;
        drop(scratch);
        Ok(())
    }
}
