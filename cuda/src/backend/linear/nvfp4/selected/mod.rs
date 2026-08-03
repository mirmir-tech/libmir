use mircuda::{DeviceBuffer, Stream, bf16};

use super::{CudaBackend, NvFp4ExpertBank};
use crate::{
    Error, Result,
    backend::linear::GatedActivation,
    kernels::{NvFp4BankView, SelectedNvFp4Gated, SelectedNvFp4Reduce, SelectedNvFp4Spec},
};

mod bucketed;
#[cfg(test)]
mod direct_down_moe;
mod grouped;
mod hybrid_gate_moe;
mod micro_moe;
mod tensor_core;
#[cfg(test)]
mod tests;

pub use bucketed::BucketedNvFp4MoeBf16;
pub(in crate::backend) use bucketed::{BucketedNvFp4Scratch, BucketedNvFp4ScratchConfig};
#[cfg(test)]
pub use direct_down_moe::DirectDownNvFp4MoeBf16;
pub use grouped::GroupedNvFp4MoeBf16;
pub use hybrid_gate_moe::HybridNvFp4MoeBf16;
pub use micro_moe::DirectNvFp4MoeBf16;
pub use tensor_core::{SelectedNvFp4LinearBf16, SelectedNvFp4TensorCoreMoeBf16};

/// Two-launch selected-expert NVFP4 MLP with device-side routing inputs.
#[derive(Debug)]
pub struct SelectedNvFp4MoeBf16 {
    gated: SelectedNvFp4Gated,
    reduce: SelectedNvFp4Reduce,
    gate: NvFp4ExpertBank,
    up: NvFp4ExpertBank,
    down: NvFp4ExpertBank,
    intermediate: DeviceBuffer<bf16>,
    output_elements: usize,
    tokens: usize,
    stream: Stream,
}

impl CudaBackend {
    /// Prepares fused gate/up and reduced down execution over contiguous banks.
    pub fn prepare_selected_nvfp4_moe_bf16(
        &self,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<SelectedNvFp4MoeBf16> {
        SelectedNvFp4MoeBf16::new(self, selected, activation, gate, up, down, 1)
    }

    pub(in crate::backend) fn prepare_batched_selected_nvfp4_moe_bf16(
        &self,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<SelectedNvFp4MoeBf16> {
        SelectedNvFp4MoeBf16::new(self, selected, activation, gate, up, down, tokens)
    }
}

impl SelectedNvFp4MoeBf16 {
    fn new(
        backend: &CudaBackend,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
        tokens: usize,
    ) -> Result<Self> {
        let expert_count_matches = gate.config.experts == down.config.experts;
        let hidden_matches = gate.config.input_features == down.config.output_features;
        let intermediate_matches = gate.config.output_features == down.config.input_features;
        if gate.config != up.config
            || !expert_count_matches
            || !hidden_matches
            || !intermediate_matches
        {
            return Err(Error::InvalidNvFp4("incompatible selected expert banks"));
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
            .ok_or(Error::InvalidNvFp4("selected intermediate size overflow"))?;
        let output_elements = tokens
            .checked_mul(spec.hidden)
            .ok_or(Error::InvalidNvFp4("selected output size overflow"))?;
        if tokens == 0 {
            return Err(Error::InvalidNvFp4("selected expert batch is empty"));
        }
        Ok(Self {
            gated: SelectedNvFp4Gated::compile(&backend.inner.compiler, spec)?,
            reduce: SelectedNvFp4Reduce::compile(&backend.inner.compiler, spec)?,
            gate,
            up,
            down,
            intermediate: backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements)?,
            output_elements,
            tokens,
            stream: backend.inner.stream.clone(),
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing_weights: &DeviceBuffer<bf16>,
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
            routing_weights,
            view(&self.down),
            output,
            self.tokens,
        )
    }

    #[must_use]
    pub const fn output_elements(&self) -> usize {
        self.output_elements
    }
}

const fn view(bank: &NvFp4ExpertBank) -> NvFp4BankView<'_> {
    NvFp4BankView {
        weight: &bank.weight,
        scales: &bank.scales,
        globals: &bank.global_scales,
        input_globals: &bank.input_scales,
        combined: &bank.combined_scales,
    }
}
