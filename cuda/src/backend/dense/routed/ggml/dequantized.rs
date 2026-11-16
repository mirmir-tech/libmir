use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, Stream, bf16};

use super::{super::super::DenseSwiGluConfig, Routing, checked, tiling};
use crate::{
    CudaBackend, CudaTensor, Error, GgmlCheckpointWeight, Result, RoutedExpertsConfig,
    backend::{
        DenseExpertWeights, dense::routed::weights::GgmlExpertWeights, linear::SelectedDenseMoeBf16,
    },
    kernels::{GgmlDequantize, GgmlRowPlacement},
};

/// Prefill chunks: the layer's banks dequantized into BF16 scratch shared by
/// every layer, then the dense grouped expert GEMM. Gate and up rows
/// interleave into one fused bank, the only layout of that GEMM's
/// tensor-core path.
pub(super) struct Dequantized {
    gate_up: Vec<GgmlDequantize>,
    down: Vec<GgmlDequantize>,
    scratch: Arc<Mutex<DeviceBuffer<bf16>>>,
    views: [DeviceBuffer<bf16>; 2],
    banks: DenseExpertWeights,
    operation: SelectedDenseMoeBf16,
    intermediate: DeviceBuffer<bf16>,
}

impl Dequantized {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        experts: RoutedExpertsConfig,
        rows: usize,
    ) -> Result<Self> {
        let (hidden, intermediate) = (config.attention.hidden_size, config.intermediate_size);
        let bank = checked(checked(experts.count, intermediate)?, hidden)?;
        let scratch = backend.ggml_dequantize_scratch(checked(bank, 3)?)?;
        let views = {
            let buffer = scratch.lock().map_err(|_| poisoned())?;
            [buffer.slice(0..2 * bank)?, buffer.slice(2 * bank..3 * bank)?]
        };
        let tensor = |name: &str, view: &DeviceBuffer<bf16>, out, input| {
            CudaTensor::from_bf16(name.into(), vec![experts.count, out, input], view.clone())
        };
        let banks = DenseExpertWeights::from_interleaved_banks(
            tensor("ggml.experts.gate_up#bf16", &views[0], 2 * intermediate, hidden),
            tensor("ggml.experts.down#bf16", &views[1], hidden, intermediate),
            experts.count,
            hidden,
            intermediate,
        );
        let compiler = backend.compiler();
        let dequantize = |rows, columns| {
            tiling(columns)
                .map(|kind| GgmlDequantize::compile(compiler, kind, rows, columns))
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            gate_up: dequantize(experts.count * intermediate, hidden)?,
            down: dequantize(experts.count * hidden, intermediate)?,
            operation: SelectedDenseMoeBf16::new(
                backend,
                rows,
                experts.top_k,
                &banks,
                config.activation.into(),
            )?,
            intermediate: backend
                .pool()
                .allocate(backend.stream(), banks.intermediate_elements(rows, experts.top_k)?)?,
            scratch,
            views,
            banks,
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
        let [gate_up, down] = &mut self.views;
        let place = |stride, offset| GgmlRowPlacement { stride, offset };
        dequantize(&self.gate_up, stream, &weights.gate, gate_up, place(2, 0))?;
        dequantize(&self.gate_up, stream, &weights.up, gate_up, place(2, 1))?;
        dequantize(&self.down, stream, &weights.down, down, place(1, 0))?;
        self.operation.execute(
            input,
            routing.selected,
            routing.weights,
            &self.banks,
            &mut self.intermediate,
            output,
        )
    }
}

fn dequantize(
    operations: &[GgmlDequantize],
    stream: &Stream,
    weight: &GgmlCheckpointWeight,
    output: &mut DeviceBuffer<bf16>,
    placement: GgmlRowPlacement,
) -> Result<()> {
    operations
        .iter()
        .find(|operation| operation.kind() == weight.kind())
        .ok_or(Error::InvalidExecutionPlan(
            "GGML expert plan has no dequantization for the bank type",
        ))?
        .execute_placed(stream, weight.blocks(), output, placement)
}

const fn poisoned() -> Error {
    Error::InvalidExecutionPlan("GGML scratch lock is poisoned")
}
