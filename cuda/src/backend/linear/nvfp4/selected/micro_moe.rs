use mircuda::{DeviceBuffer, Stream, bf16};

use super::{CudaBackend, GatedActivation, NvFp4ExpertBank, view};
use crate::{
    Error, Result,
    kernels::{
        NvFp4MicroBanks, NvFp4MicroKernels, NvFp4MicroLaunch, NvFp4MicroSpec, NvFp4MicroWorkspace,
    },
};

/// Direct small-batch NVFP4 `MoE` candidate with packed activation residency.
#[derive(Debug)]
pub struct DirectNvFp4MoeBf16 {
    kernels: NvFp4MicroKernels,
    gate: NvFp4ExpertBank,
    up: NvFp4ExpertBank,
    down: NvFp4ExpertBank,
    gate_packed: DeviceBuffer<u8>,
    up_packed: DeviceBuffer<u8>,
    gate_scales: DeviceBuffer<u8>,
    up_scales: DeviceBuffer<u8>,
    intermediate_packed: DeviceBuffer<u8>,
    intermediate_scales: DeviceBuffer<u8>,
    stream: Stream,
    tokens: usize,
    selected: usize,
    hidden: usize,
}

impl CudaBackend {
    /// Prepares a direct W4A4 candidate for isolated numerical and latency
    /// gates.
    pub fn prepare_direct_nvfp4_moe_bf16(
        &self,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<DirectNvFp4MoeBf16> {
        DirectNvFp4MoeBf16::new(self, tokens, selected, activation, gate, up, down)
    }
}

impl DirectNvFp4MoeBf16 {
    fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<Self> {
        validate(&gate, &up, &down)?;
        let experts = gate.config.experts;
        let hidden = gate.config.input_features;
        let intermediate = gate.config.output_features;
        let spec = NvFp4MicroSpec {
            experts,
            selected,
            hidden,
            intermediate,
            tokens,
            activation: activation.into(),
        };
        let groups = tokens
            .checked_mul(selected)
            .ok_or(Error::InvalidNvFp4("direct micro expert count overflow"))?;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            kernels: NvFp4MicroKernels::compile(&backend.inner.compiler, spec)?,
            gate_packed: allocate(elements(groups, hidden / 2)?)?,
            up_packed: allocate(elements(groups, hidden / 2)?)?,
            gate_scales: allocate(elements(groups, hidden / 16)?)?,
            up_scales: allocate(elements(groups, hidden / 16)?)?,
            intermediate_packed: allocate(elements(groups, intermediate / 2)?)?,
            intermediate_scales: allocate(elements(groups, intermediate / 16)?)?,
            stream: backend.inner.stream.clone(),
            gate,
            up,
            down,
            tokens,
            selected,
            hidden,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.kernels.execute(
            &self.stream,
            NvFp4MicroLaunch {
                input,
                selected,
                routing,
                banks: NvFp4MicroBanks {
                    gate: view(&self.gate),
                    up: view(&self.up),
                    down: view(&self.down),
                },
                workspace: NvFp4MicroWorkspace {
                    gate_packed: &mut self.gate_packed,
                    up_packed: &mut self.up_packed,
                    gate_scales: &mut self.gate_scales,
                    up_scales: &mut self.up_scales,
                    intermediate_packed: &mut self.intermediate_packed,
                    intermediate_scales: &mut self.intermediate_scales,
                },
                output,
            },
        )
    }

    #[must_use]
    pub fn output_elements(&self) -> usize {
        self.tokens.saturating_mul(self.hidden)
    }

    #[must_use]
    pub const fn selected_count(&self) -> usize {
        self.selected
    }
}

fn validate(gate: &NvFp4ExpertBank, up: &NvFp4ExpertBank, down: &NvFp4ExpertBank) -> Result<()> {
    let same_gate_up = gate.config == up.config;
    let transposed_down =
        (gate.config.input_features, gate.config.output_features, gate.config.experts)
            == (down.config.output_features, down.config.input_features, down.config.experts);
    let valid = same_gate_up && transposed_down;
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidNvFp4("incompatible direct micro expert banks"))
    }
}

fn elements(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(Error::InvalidNvFp4("direct micro workspace overflow"))
}
