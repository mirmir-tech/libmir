//! The layer half of Gemma 4's per-layer inputs: after the feed-forward
//! residual and before the layer scalar, a layer adds
//! `norm(projection(gelu(gate(h)) * input))` of its own input `input`
//! (Hugging Face `Gemma4TextDecoderLayer`'s per-layer input gate).

use mircuda::{DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{
    CudaTensor, DenseRole, Error, Result, RmsNormBf16,
    backend::linear::{CheckpointProjection, CheckpointProjectionWeight, ProjectionLayout},
    kernels::{ElementwiseBf16, GatedActivation},
};

#[cfg(all(test, target_os = "linux"))]
mod tests;

/// A layer's per-layer input tensors: both projections in the checkpoint's
/// storage (GGUF files quantize them like the other projections) and the
/// BF16 norm of their output.
#[derive(Clone, Debug)]
pub struct PerLayerInputWeights {
    /// `[width, hidden]`.
    pub gate: CheckpointProjectionWeight,
    /// `[hidden, width]`.
    pub projection: CheckpointProjectionWeight,
    /// `[hidden]` BF16.
    pub output_norm: CudaTensor,
}

impl PerLayerInputWeights {
    /// Checks the storage, dtype and geometry of every tensor against a
    /// layer of `hidden` features and per-layer inputs `width` wide, so a
    /// tensor no projection can read fails at load rather than in a pass.
    pub(in crate::backend::dense) fn validate(&self, hidden: usize, width: usize) -> Result<()> {
        late_bound(&self.gate)?.affine_format(1, hidden, width)?;
        late_bound(&self.projection)?.affine_format(1, width, hidden)?;
        let norm = self.output_norm.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: self.output_norm.name().into(),
            expected: "BF16",
        })?;
        if norm.len() != hidden {
            return Err(Error::InvalidQuantizedTensor {
                name: self.output_norm.name().into(),
                expected: vec![hidden],
                actual: self.output_norm.shape().to_vec(),
            });
        }
        Ok(())
    }

    /// The layouts of the gate and the projection: layers sharing one plan
    /// must store them alike.
    pub(in crate::backend) fn layouts(&self) -> [ProjectionLayout; 2] {
        [self.gate.layout(), self.projection.layout()]
    }
}

/// Layers share per-layer gate plans and bind their weights at execution,
/// which NVFP4's prepared weights cannot.
fn late_bound(weight: &CheckpointProjectionWeight) -> Result<&CheckpointProjectionWeight> {
    match weight {
        CheckpointProjectionWeight::NvFp4(_) | CheckpointProjectionWeight::NvFp4WeightOnly(_) => {
            Err(Error::UnsupportedDecoderLayer(
                "CUDA per-layer input projections cannot bind NVFP4 weights at execution".into(),
            ))
        },
        weight => Ok(weight),
    }
}

/// The per-layer input gate of a fixed number of rows, planned for the
/// storage of the (validated) weights it is built from.
pub(in crate::backend::dense) struct PerLayerGate {
    gate: CheckpointProjection,
    projection: CheckpointProjection,
    activation: ElementwiseBf16,
    norm: RmsNormBf16,
    gated: DeviceBuffer<bf16>,
    activated: DeviceBuffer<bf16>,
    stream: Stream,
}

impl PerLayerGate {
    pub(in crate::backend::dense) fn new(
        backend: &CudaBackend,
        rows: usize,
        [hidden, width]: [usize; 2],
        epsilon: f32,
        weights: &PerLayerInputWeights,
    ) -> Result<Self> {
        let elements = super::scratch::elements(rows, width)?;
        let allocate = || backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements);
        let projection = |input, output, role, weight| {
            CheckpointProjection::new(backend, rows, input, output, role, weight)
        };
        Ok(Self {
            gate: projection(hidden, width, DenseRole::DenseGateUp, &weights.gate)?,
            projection: projection(width, hidden, DenseRole::DenseDown, &weights.projection)?,
            activation: ElementwiseBf16::compile(&backend.inner.compiler, elements)?,
            norm: RmsNormBf16::new(backend, rows, hidden, epsilon)?,
            gated: allocate()?,
            activated: allocate()?,
            stream: backend.inner.stream.clone(),
        })
    }

    /// Writes `norm(projection(gelu(gate(hidden)) * input))` into
    /// `normalized`, projecting through `projected`.
    pub(in crate::backend::dense) fn execute(
        &mut self,
        hidden: &DeviceBuffer<bf16>,
        weights: &PerLayerInputWeights,
        input: &DeviceBuffer<bf16>,
        projected: &mut DeviceBuffer<bf16>,
        normalized: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.gate.execute_with(hidden, &weights.gate, &mut self.gated)?;
        self.activation.gated(
            &self.stream,
            &self.gated,
            input,
            &mut self.activated,
            GatedActivation::GeluTanh,
        )?;
        self.projection.execute_with(&self.activated, &weights.projection, projected)?;
        self.norm.execute(projected, &weights.output_norm, normalized)
    }
}
