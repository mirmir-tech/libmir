use mircuda::{DeviceBuffer, Stream, bf16};

use super::{CudaBackend, NvFp4ExpertBank, grouped::GroupedNvFp4PairBf16, view};
use crate::{
    Error, GatedActivation, Result,
    kernels::{
        NvFp4MicroDownKernels, NvFp4MicroDownLaunch, NvFp4MicroDownWorkspace, NvFp4MicroSpec,
    },
};

/// Tensor Core gate/up followed by direct `SM12x` activation and down
/// reduction.
#[derive(Debug)]
pub struct DirectDownNvFp4MoeBf16 {
    gate_up: GroupedNvFp4PairBf16,
    down_kernels: NvFp4MicroDownKernels,
    down: NvFp4ExpertBank,
    gate_output: DeviceBuffer<bf16>,
    up_output: DeviceBuffer<bf16>,
    packed: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    stream: Stream,
    tokens: usize,
    selected: usize,
    hidden: usize,
}

impl CudaBackend {
    /// Prepares the hybrid W4A4 candidate for explicit numerical and latency
    /// gates.
    pub fn prepare_direct_down_nvfp4_moe_bf16(
        &self,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<DirectDownNvFp4MoeBf16> {
        DirectDownNvFp4MoeBf16::new(self, tokens, selected, activation, gate, up, down)
    }
}

impl DirectDownNvFp4MoeBf16 {
    #[allow(clippy::too_many_arguments)]
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
            .ok_or(Error::InvalidNvFp4("hybrid expert count overflow"))?;
        let intermediate_elements = elements(groups, intermediate)?;
        Ok(Self {
            gate_up: GroupedNvFp4PairBf16::new(backend, tokens, selected, gate, up)?,
            down_kernels: NvFp4MicroDownKernels::compile(&backend.inner.compiler, spec)?,
            gate_output: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, intermediate_elements)?,
            up_output: backend.inner.pool.allocate(&backend.inner.stream, intermediate_elements)?,
            packed: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, elements(groups, intermediate / 2)?)?,
            scales: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, elements(groups, intermediate / 16)?)?,
            stream: backend.inner.stream.clone(),
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
        self.gate_up
            .execute(input, selected, &mut self.gate_output, &mut self.up_output)?;
        self.down_kernels.execute(
            &self.stream,
            NvFp4MicroDownLaunch {
                gate: &self.gate_output,
                up: &self.up_output,
                selected,
                routing,
                down: view(&self.down),
                workspace: NvFp4MicroDownWorkspace {
                    packed: &mut self.packed,
                    scales: &mut self.scales,
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
    let valid = gate.config == up.config
        && (gate.config.input_features, gate.config.output_features, gate.config.experts)
            == (down.config.output_features, down.config.input_features, down.config.experts);
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidNvFp4("incompatible hybrid expert banks"))
    }
}

fn elements(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(Error::InvalidNvFp4("hybrid workspace overflow"))
}
