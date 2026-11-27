pub(super) mod scratch;

use std::sync::{Arc, Mutex};

use scratch::DenseScratch;

#[cfg(all(test, target_os = "linux"))]
mod tests;
use mircuda::{DeviceBuffer, Stream, bf16};
use models::weights::{DenseFeedForwardBindings, TensorBinding};

use crate::{
    CudaBackend, CudaTensorSet, DenseRole, Error, GatedActivation, Result,
    backend::linear::{CheckpointProjection, CheckpointProjectionWeight},
    kernels::{ElementwiseBf16, ReluSquaredBf16},
};

#[derive(Clone, Debug)]
pub(in crate::backend) struct DenseFeedForward {
    backend: CudaBackend,
    hidden: usize,
    intermediate: usize,
    mlp: DenseMlp,
    down: CheckpointProjectionWeight,
}

/// The projections before `down` and the activation joining them.
#[derive(Clone, Debug)]
enum DenseMlp {
    Gated {
        gate: Box<CheckpointProjectionWeight>,
        up: CheckpointProjectionWeight,
        activation: GatedActivation,
    },
    /// Nemotron-H's ungated `relu(up(x))²`.
    ReluSquared { up: CheckpointProjectionWeight },
}

impl DenseFeedForward {
    /// A gated MLP between `hidden` and `intermediate` features.
    pub(in crate::backend) fn load(
        backend: &CudaBackend,
        widths: (usize, usize),
        activation: GatedActivation,
        tensors: &CudaTensorSet,
        bindings: DenseFeedForwardBindings<'_>,
    ) -> Result<Self> {
        let load =
            |binding| CheckpointProjectionWeight::load_binding_prepared(backend, tensors, binding);
        let mlp = DenseMlp::Gated {
            gate: Box::new(load(bindings.gate)?),
            up: load(bindings.up)?,
            activation,
        };
        Self::validated(backend, widths, mlp, load(bindings.down)?)
    }

    /// An ungated `relu²` MLP of `up` and `down` between `hidden` and
    /// `intermediate` features.
    pub(in crate::backend) fn load_relu_squared(
        backend: &CudaBackend,
        widths: (usize, usize),
        tensors: &CudaTensorSet,
        (up, down): (&TensorBinding, &TensorBinding),
    ) -> Result<Self> {
        let load =
            |binding| CheckpointProjectionWeight::load_binding_prepared(backend, tensors, binding);
        Self::validated(backend, widths, DenseMlp::ReluSquared { up: load(up)? }, load(down)?)
    }

    fn validated(
        backend: &CudaBackend,
        (hidden, intermediate): (usize, usize),
        mlp: DenseMlp,
        down: CheckpointProjectionWeight,
    ) -> Result<Self> {
        match &mlp {
            DenseMlp::Gated { gate, up, .. } => {
                gate.affine_format(1, hidden, intermediate)?;
                up.affine_format(1, hidden, intermediate)?;
            },
            DenseMlp::ReluSquared { up } => {
                up.affine_format(1, hidden, intermediate)?;
            },
        }
        down.affine_format(1, intermediate, hidden)?;
        Ok(Self {
            backend: backend.clone(),
            hidden,
            intermediate,
            mlp,
            down,
        })
    }

    pub(super) fn prepare(&self, tokens: usize) -> Result<DenseExecution> {
        let elements = tokens
            .checked_mul(self.intermediate)
            .ok_or(Error::InvalidDecoderKernel("dense intermediate size overflow"))?;
        let projection = |input, output, role, weight| {
            CheckpointProjection::new(&self.backend, tokens, input, output, role, weight)
        };
        let up =
            |weight| projection(self.hidden, self.intermediate, DenseRole::DenseGateUp, weight);
        let mlp = match &self.mlp {
            DenseMlp::Gated { gate, up: up_weight, activation } => MlpExecution::Gated {
                gate: Box::new(up(gate)?),
                up: up(up_weight)?,
                activation: *activation,
                elementwise: ElementwiseBf16::compile(&self.backend.inner.compiler, elements)?,
            },
            DenseMlp::ReluSquared { up: up_weight } => MlpExecution::ReluSquared {
                up: up(up_weight)?,
                activation: ReluSquaredBf16::compile(&self.backend.inner.compiler)?,
                elements,
            },
        };
        Ok(DenseExecution {
            backend: self.backend.clone(),
            mlp,
            down: projection(self.intermediate, self.hidden, DenseRole::DenseDown, &self.down)?,
            scratch: self.backend.inner.dense_scratch.acquire(&self.backend, elements)?,
        })
    }
}

#[derive(Debug)]
enum MlpExecution {
    Gated {
        gate: Box<CheckpointProjection>,
        up: CheckpointProjection,
        activation: GatedActivation,
        elementwise: ElementwiseBf16,
    },
    ReluSquared {
        up: CheckpointProjection,
        activation: ReluSquaredBf16,
        elements: usize,
    },
}

#[derive(Debug)]
pub(in crate::backend) struct DenseExecution {
    backend: CudaBackend,
    mlp: MlpExecution,
    down: CheckpointProjection,
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
        // the lock through the whole MLP so another submission cannot
        // interleave.
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("dense scratch lock is poisoned"))?;
        let DenseScratch { gate_output, up_output, activated } = &mut *scratch;
        let stream = &self.backend.inner.stream;
        match &mut self.mlp {
            MlpExecution::Gated { gate, up, activation, elementwise } => {
                gate.execute(input, gate_output)?;
                up.execute(input, up_output)?;
                elementwise.gated(
                    stream,
                    gate_output,
                    up_output,
                    activated,
                    (*activation).into(),
                )?;
            },
            MlpExecution::ReluSquared { up, activation, elements } => {
                up.execute(input, up_output)?;
                activation.execute(stream, up_output, activated, *elements)?;
            },
        }
        self.down.execute(activated, output)?;
        drop(scratch);
        Ok(())
    }
}
