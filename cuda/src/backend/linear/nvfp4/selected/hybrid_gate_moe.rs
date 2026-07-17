use mircuda::{DeviceBuffer, Stream, bf16};

use super::{CudaBackend, NvFp4ExpertBank, grouped::GroupedNvFp4LinearBf16, view};
use crate::{
    Error, GatedActivation, Result,
    kernels::{
        ElementwiseBf16, NvFp4MicroBanks, NvFp4MicroGateLaunch, NvFp4MicroGateWorkspace,
        NvFp4MicroKernels, NvFp4MicroSpec,
    },
};

/// Direct gate/up followed by Tensor Core down projection and weighted
/// reduction.
#[derive(Debug)]
pub struct HybridNvFp4MoeBf16 {
    gate_kernels: NvFp4MicroKernels,
    gate: NvFp4ExpertBank,
    up: NvFp4ExpertBank,
    down_bank: NvFp4ExpertBank,
    down: GroupedNvFp4LinearBf16,
    reduce: ElementwiseBf16,
    gate_packed: DeviceBuffer<u8>,
    up_packed: DeviceBuffer<u8>,
    gate_scales: DeviceBuffer<u8>,
    up_scales: DeviceBuffer<u8>,
    down_output: DeviceBuffer<bf16>,
    stream: Stream,
    tokens: usize,
    selected: usize,
    hidden: usize,
}

impl CudaBackend {
    /// Prepares direct gate/up with a Tensor Core down tail for explicit gates.
    pub fn prepare_hybrid_gate_nvfp4_moe_bf16(
        &self,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<HybridNvFp4MoeBf16> {
        HybridNvFp4MoeBf16::new(self, tokens, selected, activation, gate, up, down)
    }
}

impl HybridNvFp4MoeBf16 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down_bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        validate(&gate, &up, &down_bank)?;
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
            .ok_or(Error::InvalidNvFp4("hybrid gate expert count overflow"))?;
        let down = GroupedNvFp4LinearBf16::new(backend, tokens, selected, down_bank.clone())?;
        let allocate_bytes =
            |count| backend.inner.pool.allocate::<u8>(&backend.inner.stream, count);
        Ok(Self {
            gate_kernels: NvFp4MicroKernels::compile(&backend.inner.compiler, spec)?,
            gate_packed: allocate_bytes(elements(groups, hidden / 2)?)?,
            up_packed: allocate_bytes(elements(groups, hidden / 2)?)?,
            gate_scales: allocate_bytes(elements(groups, hidden / 16)?)?,
            up_scales: allocate_bytes(elements(groups, hidden / 16)?)?,
            down_output: backend
                .inner
                .pool
                .allocate::<bf16>(&backend.inner.stream, elements(groups, hidden)?)?,
            reduce: ElementwiseBf16::compile(&backend.inner.compiler, hidden)?,
            stream: backend.inner.stream.clone(),
            gate,
            up,
            down_bank,
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
        let scale_stride = self.down.scale_stride()?;
        {
            let (packed, scales) = self.down.quantized_workspace();
            self.gate_kernels.execute_gate(
                &self.stream,
                NvFp4MicroGateLaunch {
                    input,
                    selected,
                    banks: NvFp4MicroBanks {
                        gate: view(&self.gate),
                        up: view(&self.up),
                        down: view(&self.down_bank),
                    },
                    workspace: NvFp4MicroGateWorkspace {
                        gate_packed: &mut self.gate_packed,
                        up_packed: &mut self.up_packed,
                        gate_scales: &mut self.gate_scales,
                        up_scales: &mut self.up_scales,
                        output_packed: packed,
                        output_scales: scales,
                    },
                    output_scale_stride: scale_stride,
                },
            )?;
        }
        self.down.execute_prepared(selected, &mut self.down_output)?;
        self.reduce.weighted_reduce_batch(
            &self.stream,
            &self.down_output,
            routing,
            output,
            self.selected,
            self.tokens,
        )
    }

    #[must_use]
    pub fn output_elements(&self) -> usize {
        self.tokens.saturating_mul(self.hidden)
    }
}

fn validate(gate: &NvFp4ExpertBank, up: &NvFp4ExpertBank, down: &NvFp4ExpertBank) -> Result<()> {
    let valid = gate.config == up.config
        && (gate.config.input_features, gate.config.output_features, gate.config.experts)
            == (down.config.output_features, down.config.input_features, down.config.experts);
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidNvFp4("incompatible hybrid gate expert banks"))
    }
}

fn elements(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(Error::InvalidNvFp4("hybrid gate workspace overflow"))
}
