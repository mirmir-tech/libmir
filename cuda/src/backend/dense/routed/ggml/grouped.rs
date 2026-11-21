use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, Stream, bf16};

use super::{GgmlExpertGeometry, Routing, checked, tiling};
use crate::{
    CudaBackend, Error, GgmlCheckpointWeight, Result,
    backend::dense::routed::weights::GgmlExpertWeights,
    kernels::{
        DenseGateUpLayout, GgmlActivations, GgmlGroupedMmq, GgmlMmqGroups, GgmlMmqInput,
        GgmlMmqPlacement, GgmlRouteCombine, SelectedDenseDispatch, SelectedDenseMoe,
        SelectedDenseMoeSpec,
    },
};

/// Prefill chunks: the selections sorted by expert, their tokens gathered
/// into quantized rows, then grouped int8 GEMMs straight over the GGML
/// banks. Gate and up land interleaved in one buffer for the shared
/// activation kernel; each token then sums its down rows weighted by
/// routing. Every scratch is shared by all layers: they run in order on the
/// stream.
pub(super) struct Grouped {
    operation: SelectedDenseMoe,
    gate_up: GgmlGroupedMmq,
    down: GgmlGroupedMmq,
    combine: GgmlRouteCombine,
    dispatch: Dispatch,
    scratch: Arc<Mutex<DeviceBuffer<bf16>>>,
    /// Down rows, gate and up interleaved, and their activation.
    views: [DeviceBuffer<bf16>; 3],
    hidden_activations: Arc<Mutex<GgmlActivations>>,
    intermediate_activations: Arc<Mutex<GgmlActivations>>,
    hidden: usize,
    intermediate: usize,
    routes: usize,
}

struct Dispatch {
    counts: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    cursors: DeviceBuffer<u32>,
    assignments: DeviceBuffer<u32>,
    experts: DeviceBuffer<u32>,
}

impl Grouped {
    pub(super) fn new(
        backend: &CudaBackend,
        geometry: GgmlExpertGeometry,
        rows: usize,
    ) -> Result<Self> {
        let (hidden, intermediate, experts) =
            (geometry.hidden, geometry.intermediate, geometry.experts);
        let routes = checked(rows, experts.top_k)?;
        let operation = SelectedDenseMoe::compile(
            backend.compiler(),
            SelectedDenseMoeSpec {
                tokens: rows,
                input_features: hidden,
                output_features: intermediate,
                expert_count: experts.count,
                selected_count: experts.top_k,
                gate_up_layout: DenseGateUpLayout::FusedInterleaved,
                gate_transposed: false,
                up_transposed: false,
                down_transposed: false,
                gate_bias: false,
                up_bias: false,
                down_bias: false,
                activation: geometry.activation.into(),
            },
        )?;
        let (pool, stream) = (backend.pool(), backend.stream());
        let compile = |rows, columns| {
            let kinds = tiling(columns).collect::<Vec<_>>();
            GgmlGroupedMmq::compile(
                backend.compiler(),
                pool,
                stream,
                &kinds,
                experts.count,
                rows,
                columns,
                routes,
            )
        };
        let widths = [hidden, 2 * intermediate, intermediate];
        let elements = widths.iter().try_fold(0, |total: usize, &width| {
            total.checked_add(checked(routes, width)?).ok_or(overflow())
        })?;
        let scratch = backend.ggml_dequantize_scratch(elements)?;
        let views = {
            let buffer = scratch.lock().map_err(|_| poisoned())?;
            let mut start = 0;
            widths.map(|width| {
                let end = start + routes * width;
                let view = buffer.slice(start..end);
                start = end;
                view
            })
        };
        let [projected, fused, activated] = views;
        let allocate = |elements| pool.allocate::<u32>(stream, elements);
        Ok(Self {
            combine: GgmlRouteCombine::compile(
                backend.compiler(),
                allocate(routes)?,
                rows,
                experts.top_k,
                hidden,
                experts.count,
            )?,
            gate_up: compile(intermediate, hidden)?,
            down: compile(hidden, intermediate)?,
            dispatch: Dispatch {
                counts: allocate(experts.count)?,
                offsets: allocate(experts.count)?,
                cursors: allocate(experts.count)?,
                assignments: allocate(routes)?,
                experts: allocate(routes)?,
            },
            operation,
            views: [projected?, fused?, activated?],
            scratch,
            hidden_activations: backend.ggml_activations(routes, hidden)?,
            intermediate_activations: backend.ggml_activations(routes, intermediate)?,
            hidden,
            intermediate,
            routes,
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
        let _scratch = self.scratch.lock().map_err(|_| poisoned())?;
        let mut hidden_activations = self.hidden_activations.lock().map_err(|_| poisoned())?;
        let mut intermediate_activations =
            self.intermediate_activations.lock().map_err(|_| poisoned())?;
        let [projected, fused, activated] = &mut self.views;
        let mut dispatch = SelectedDenseDispatch {
            counts: &mut self.dispatch.counts,
            offsets: &mut self.dispatch.offsets,
            cursors: &mut self.dispatch.cursors,
            assignments: &mut self.dispatch.assignments,
            experts: &mut self.dispatch.experts,
        };
        self.operation.prepare_expert_major(stream, routing.selected, &mut dispatch)?;
        let groups = GgmlMmqGroups {
            offsets: &*dispatch.offsets,
            counts: &*dispatch.counts,
        };
        let gathered = GgmlMmqInput::Gathered {
            tokens: input,
            assignments: &*dispatch.assignments,
            top_k: routing.top_k,
        };
        self.gate_up
            .prepare(stream, gathered, self.routes, groups, &mut hidden_activations)?;
        let interleaved = |offset| GgmlMmqPlacement {
            ld: 2 * self.intermediate,
            stride: 2,
            offset,
        };
        let gate_up = &self.gate_up;
        project(
            gate_up,
            stream,
            &hidden_activations,
            groups,
            &weights.gate,
            fused,
            interleaved(0),
        )?;
        project(gate_up, stream, &hidden_activations, groups, &weights.up, fused, interleaved(1))?;
        self.operation.activate_compact(stream, fused, activated)?;
        let activated = GgmlMmqInput::Rows(activated);
        self.down
            .prepare(stream, activated, self.routes, groups, &mut intermediate_activations)?;
        let rows = GgmlMmqPlacement { ld: self.hidden, stride: 1, offset: 0 };
        project(
            &self.down,
            stream,
            &intermediate_activations,
            groups,
            &weights.down,
            projected,
            rows,
        )?;
        self.combine.execute(
            stream,
            projected,
            &*dispatch.assignments,
            routing.selected,
            routing.weights,
            output,
        )
    }
}

fn project(
    operation: &GgmlGroupedMmq,
    stream: &Stream,
    activations: &GgmlActivations,
    groups: GgmlMmqGroups<'_>,
    weight: &GgmlCheckpointWeight,
    output: &mut DeviceBuffer<bf16>,
    placement: GgmlMmqPlacement,
) -> Result<()> {
    operation.multiply(
        stream,
        activations,
        groups,
        weight.kind(),
        weight.blocks(),
        output,
        placement,
    )
}

const fn poisoned() -> Error {
    Error::InvalidExecutionPlan("GGML grouped expert scratch lock is poisoned")
}

const fn overflow() -> Error {
    Error::InvalidDecoderKernel("GGML grouped expert scratch size overflow")
}
