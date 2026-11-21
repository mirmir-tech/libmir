use mircuda::{DeviceBuffer, Stream, bf16};

use super::{GgmlExpertGeometry, Routing, checked, tiling};
use crate::{
    CudaBackend, Error, GatedActivation, Result,
    backend::dense::routed::weights::GgmlExpertWeights,
    kernels::{
        GgmlActivations, GgmlExpertSpec, GgmlGatheredLinear, GgmlKernelType, GgmlSelections,
        PackedGatedBf16,
    },
};

/// Decode batches: one int8 GEMV per (token, expert) selection over the
/// stacked banks, then the routing-weighted sum per token.
pub(super) struct Gathered {
    gate_up: Vec<GgmlGatheredLinear>,
    down: Vec<GgmlGatheredLinear>,
    input: GgmlActivations,
    intermediate: GgmlActivations,
    gate: DeviceBuffer<bf16>,
    up: DeviceBuffer<bf16>,
    activated: DeviceBuffer<bf16>,
    partial: DeviceBuffer<bf16>,
    activation: PackedGatedBf16,
    kind: GatedActivation,
    rows: usize,
}

impl Gathered {
    pub(super) fn new(
        backend: &CudaBackend,
        geometry: GgmlExpertGeometry,
        rows: usize,
    ) -> Result<Self> {
        let (hidden, intermediate, experts) =
            (geometry.hidden, geometry.intermediate, geometry.experts);
        let selections = checked(rows, experts.top_k)?;
        let compiler = backend.compiler();
        let kinds = |rows, columns| {
            tiling(columns)
                .map(|kind| {
                    let spec = GgmlExpertSpec {
                        kind,
                        experts: experts.count,
                        rows,
                        columns,
                    };
                    GgmlGatheredLinear::compile(compiler, spec)
                })
                .collect::<Result<Vec<_>>>()
        };
        let allocate = |elements| backend.pool().allocate(backend.stream(), elements);
        Ok(Self {
            gate_up: kinds(intermediate, hidden)?,
            down: kinds(hidden, intermediate)?,
            input: GgmlActivations::new(backend.pool(), backend.stream(), rows, hidden)?,
            intermediate: GgmlActivations::new(
                backend.pool(),
                backend.stream(),
                selections,
                intermediate,
            )?,
            gate: allocate(checked(selections, intermediate)?)?,
            up: allocate(checked(selections, intermediate)?)?,
            activated: allocate(checked(selections, intermediate)?)?,
            partial: allocate(checked(selections, hidden)?)?,
            activation: PackedGatedBf16::compile(compiler, selections, intermediate)?,
            kind: geometry.activation,
            rows,
        })
    }

    pub(super) fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        routing: &Routing<'_>,
        weights: &GgmlExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let (gate_bank, up_bank, down_bank) = (weights.gate(), weights.up(), weights.down());
        let gate = find(&self.gate_up, gate_bank.kind())?;
        let up = find(&self.gate_up, up_bank.kind())?;
        let down = find(&self.down, down_bank.kind())?;
        let count = checked(self.rows, routing.top_k)?;
        gate.quantize(stream, input, &mut self.input, self.rows)?;
        let shared = GgmlSelections {
            selected: routing.selected,
            count,
            group: routing.top_k,
        };
        gate.multiply(stream, &self.input, gate_bank.operand(), &shared, &mut self.gate)?;
        up.multiply(stream, &self.input, up_bank.operand(), &shared, &mut self.up)?;
        self.activation.execute_separate(
            stream,
            &self.gate,
            &self.up,
            &mut self.activated,
            self.kind.into(),
        )?;
        down.quantize(stream, &self.activated, &mut self.intermediate, count)?;
        let own = GgmlSelections {
            selected: routing.selected,
            count,
            group: 1,
        };
        down.multiply(stream, &self.intermediate, down_bank.operand(), &own, &mut self.partial)?;
        down.combine(stream, &self.partial, routing.weights, output, routing.top_k)
    }
}

fn find(operations: &[GgmlGatheredLinear], kind: GgmlKernelType) -> Result<&GgmlGatheredLinear> {
    operations
        .iter()
        .find(|operation| operation.spec().kind == kind)
        .ok_or(Error::InvalidExecutionPlan("GGML expert plan has no kernel for the bank type"))
}
