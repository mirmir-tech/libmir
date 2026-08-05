use mircuda::{DeviceBuffer, Stream, bf16};

use super::{CudaBackend, NvFp4ExpertBank, view};
use crate::{
    Error, Result,
    backend::linear::GatedActivation,
    kernels::{
        SelectedNvFp4Spec, SelectedNvFp4TiledGated, SelectedNvFp4TiledReduce,
        SelectedNvFp4TiledRows,
    },
};

/// Tiled selected-expert W4A16 MLP candidate measured by the CUDA autotuner.
#[derive(Debug)]
pub struct TiledSelectedNvFp4MoeBf16 {
    gated: SelectedNvFp4TiledGated,
    reduce: SelectedNvFp4TiledReduce,
    gate: NvFp4ExpertBank,
    up: NvFp4ExpertBank,
    down: NvFp4ExpertBank,
    intermediate: DeviceBuffer<bf16>,
    tokens: usize,
    stream: Stream,
}

impl TiledSelectedNvFp4MoeBf16 {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        rows: SelectedNvFp4TiledRows,
        banks: [NvFp4ExpertBank; 3],
    ) -> Result<Self> {
        let [gate, up, down] = banks;
        let expert_count_matches = gate.config.experts == down.config.experts;
        let hidden_matches = gate.config.input_features == down.config.output_features;
        let intermediate_matches = gate.config.output_features == down.config.input_features;
        if tokens == 0
            || gate.config != up.config
            || !expert_count_matches
            || !hidden_matches
            || !intermediate_matches
        {
            return Err(Error::InvalidNvFp4("incompatible tiled selected expert banks"));
        }
        let spec = SelectedNvFp4Spec {
            experts: gate.config.experts,
            selected,
            hidden: gate.config.input_features,
            intermediate: gate.config.output_features,
            activation: activation.into(),
        };
        let elements = tokens
            .checked_mul(selected)
            .and_then(|value| value.checked_mul(spec.intermediate))
            .ok_or(Error::InvalidNvFp4("tiled selected intermediate size overflow"))?;
        Ok(Self {
            gated: SelectedNvFp4TiledGated::compile(&backend.inner.compiler, spec, rows)?,
            reduce: SelectedNvFp4TiledReduce::compile(&backend.inner.compiler, spec, rows)?,
            gate,
            up,
            down,
            intermediate: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            tokens,
            stream: backend.inner.stream.clone(),
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.gated.execute_batch(
            &self.stream,
            input,
            selected,
            view(&self.gate),
            view(&self.up),
            &mut self.intermediate,
            self.tokens,
        )?;
        self.reduce.execute_batch(
            &self.stream,
            &self.intermediate,
            selected,
            routing,
            view(&self.down),
            output,
            self.tokens,
        )
    }
}
