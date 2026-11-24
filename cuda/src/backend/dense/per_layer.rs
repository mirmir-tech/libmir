//! The layer half of Gemma 4's per-layer inputs: after the feed-forward
//! residual and before the layer scalar, a layer adds
//! `norm(projection(gelu(gate(h)) * input))` of its own input `input`
//! (Hugging Face `Gemma4TextDecoderLayer`'s per-layer input gate).

use mircuda::{DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{
    Bf16Projection, CudaTensor, DensePlanRequest, DenseRole, ExecutionPhase, Result, RmsNormBf16,
    kernels::{ElementwiseBf16, GatedActivation},
};

/// A layer's per-layer input tensors: BF16 projections and the norm of
/// their output.
#[derive(Clone, Copy)]
pub struct PerLayerInputWeights<'a> {
    /// `[width, hidden]`.
    pub gate: &'a CudaTensor,
    /// `[hidden, width]`.
    pub projection: &'a CudaTensor,
    pub output_norm: &'a CudaTensor,
}

/// Device-resident [`PerLayerInputWeights`] retained by a layer.
#[derive(Clone)]
pub(in crate::backend::dense) struct PerLayerInputOwned {
    gate: CudaTensor,
    projection: CudaTensor,
    output_norm: CudaTensor,
}

impl PerLayerInputOwned {
    pub(in crate::backend::dense) fn new(weights: PerLayerInputWeights<'_>) -> Self {
        Self {
            gate: weights.gate.clone(),
            projection: weights.projection.clone(),
            output_norm: weights.output_norm.clone(),
        }
    }

    pub(in crate::backend::dense) const fn borrow(&self) -> PerLayerInputWeights<'_> {
        PerLayerInputWeights {
            gate: &self.gate,
            projection: &self.projection,
            output_norm: &self.output_norm,
        }
    }
}

/// The per-layer input gate of a fixed number of rows.
pub(in crate::backend::dense) struct PerLayerGate {
    gate: Bf16Projection,
    projection: Bf16Projection,
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
    ) -> Result<Self> {
        let request = |role, input_features, output_features| DensePlanRequest {
            phase: if rows == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            role,
            tokens: rows,
            input_features,
            output_features,
        };
        let elements = super::scratch::elements(rows, width)?;
        let allocate = || backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements);
        Ok(Self {
            gate: backend.prepare_bf16_projection(request(
                DenseRole::DenseGateUp,
                hidden,
                width,
            ))?,
            projection: backend.prepare_bf16_projection(request(
                DenseRole::DenseDown,
                width,
                hidden,
            ))?,
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
        weights: PerLayerInputWeights<'_>,
        input: &DeviceBuffer<bf16>,
        projected: &mut DeviceBuffer<bf16>,
        normalized: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.gate.execute(hidden, weights.gate, &mut self.gated)?;
        self.activation.gated(
            &self.stream,
            &self.gated,
            input,
            &mut self.activated,
            GatedActivation::GeluTanh,
        )?;
        self.projection.execute(&self.activated, weights.projection, projected)?;
        self.norm.execute(projected, weights.output_norm, normalized)
    }
}
