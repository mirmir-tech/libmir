//! The shared expert beside routed experts (GLM-4.5): a gated MLP every
//! token runs on the same normalized input, added to the routed output.

use mircuda::{DeviceBuffer, Stream, bf16};

use super::weights::SharedExpertWeights;
use crate::{
    CudaBackend, DenseRole, GatedActivation, Result,
    backend::linear::CheckpointProjection,
    kernels::{ElementwiseBf16, PackedGatedBf16},
};

/// Planned from one layer's weights; every layer executes it with its own.
pub(super) struct SharedExpert {
    gate: CheckpointProjection,
    up: CheckpointProjection,
    down: CheckpointProjection,
    activation: PackedGatedBf16,
    kind: GatedActivation,
    gated: [DeviceBuffer<bf16>; 3],
    /// The routed experts' output, which the shared expert's joins.
    pub(super) routed: DeviceBuffer<bf16>,
    output: DeviceBuffer<bf16>,
    sum: ElementwiseBf16,
    stream: Stream,
}

impl SharedExpert {
    pub(super) fn new(
        backend: &CudaBackend,
        rows: usize,
        (hidden, intermediate): (usize, usize),
        kind: GatedActivation,
        weights: &SharedExpertWeights,
    ) -> Result<Self> {
        let projection = |input, output, role, weight| {
            CheckpointProjection::new(backend, rows, input, output, role, weight)
        };
        let buffer = |elements: usize| backend.pool().allocate(backend.stream(), elements);
        let (hidden_elements, gated_elements) = (
            super::super::scratch::elements(rows, hidden)?,
            super::super::scratch::elements(rows, intermediate)?,
        );
        Ok(Self {
            gate: projection(hidden, intermediate, DenseRole::DenseGateUp, &weights.gate)?,
            up: projection(hidden, intermediate, DenseRole::DenseGateUp, &weights.up)?,
            down: projection(intermediate, hidden, DenseRole::DenseDown, &weights.down)?,
            activation: PackedGatedBf16::compile(backend.compiler(), rows, intermediate)?,
            kind,
            gated: [buffer(gated_elements)?, buffer(gated_elements)?, buffer(gated_elements)?],
            routed: buffer(hidden_elements)?,
            output: buffer(hidden_elements)?,
            sum: ElementwiseBf16::compile(backend.compiler(), hidden_elements)?,
            stream: backend.stream().clone(),
        })
    }

    /// Writes the routed output in [`Self::routed`] plus the shared expert of
    /// `input` into `output`.
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: &SharedExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let [gate, up, activated] = &mut self.gated;
        self.gate.execute_with(input, &weights.gate, gate)?;
        self.up.execute_with(input, &weights.up, up)?;
        self.activation
            .execute_separate(&self.stream, gate, up, activated, self.kind.into())?;
        self.down.execute_with(activated, &weights.down, &mut self.output)?;
        self.sum.add(&self.stream, &self.routed, &self.output, output)
    }
}
