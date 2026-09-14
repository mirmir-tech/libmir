use std::sync::{Arc, Mutex, atomic::AtomicBool};

use super::Stream;
use crate::engine::{
    PagedArenaPool, Result, compiled::CompiledGraphs, gate_up_tuning::MetalTuner, kernels::Kernels,
};

impl Stream {
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.native.synchronize()?)
    }

    #[cfg(test)]
    pub(crate) const fn history_budget(
        &self,
    ) -> &Arc<crate::engine::persistent_history::budget::Budget> {
        &self.history_budget
    }

    #[cfg(test)]
    pub(crate) fn set_decode_reservation(&mut self, plan: crate::config::DecodeReservation) {
        Arc::make_mut(&mut self.config).cache.decode_reservation = plan;
    }

    #[cfg(test)]
    pub(crate) fn set_history_batching(&mut self, plan: crate::config::HistoryBatching) {
        Arc::make_mut(&mut self.config).diagnostics.history_batching = plan;
    }

    #[cfg(test)]
    pub(super) fn experimental_recurrence(
        &self,
        inputs: [&mirtal::Array; 6],
    ) -> Result<[mirtal::Array; 2]> {
        use crate::engine::kernels::gated_delta::experiment::{self, Layout};
        let result = self.kernels.gdn_experiment.run(&self.native, inputs, Layout::Packed)?;
        if inputs[0].shape()?.dimensions()[3] == 128
            && inputs[2].shape()?.dimensions()[3].is_multiple_of(8)
        {
            experiment::record();
        }
        Ok(result)
    }

    #[cfg(test)]
    pub(crate) fn set_gdn_execution(&mut self, plan: crate::config::GdnExecution) {
        Arc::make_mut(&mut self.config).diagnostics.gdn_execution = plan;
    }

    #[cfg(test)]
    pub(crate) fn set_key_value_projection(&mut self, plan: crate::config::KeyValueProjection) {
        Arc::make_mut(&mut self.config).diagnostics.key_value_projection = plan;
    }

    #[cfg(test)]
    pub(crate) fn set_rope_batching(&mut self, plan: crate::config::RopeBatching) {
        Arc::make_mut(&mut self.config).diagnostics.rope_batching = plan;
    }

    #[cfg(test)]
    pub(crate) fn set_router_precision_for_test(&mut self, mode: crate::config::RouterPrecision) {
        Arc::make_mut(&mut self.config).diagnostics.router_precision = mode;
    }

    #[cfg(test)]
    pub(crate) fn set_moe_decode_probe(&mut self, probe: Option<crate::config::MoeDecodeProbe>) {
        Arc::make_mut(&mut self.config).diagnostics.moe_decode_probe = probe;
    }

    #[cfg(test)]
    pub(crate) fn set_profile_components(&mut self, enabled: bool) {
        Arc::make_mut(&mut self.config).diagnostics.profile_components = enabled;
    }

    #[cfg(test)]
    pub(crate) fn set_moe_prefill(&mut self, mode: crate::config::MoePrefill) {
        Arc::make_mut(&mut self.config).diagnostics.moe_prefill = mode;
    }

    #[cfg(test)]
    pub(crate) fn set_attention_candidate(
        &mut self,
        candidate: Option<crate::engine::BatchAttentionExecution>,
    ) {
        Arc::make_mut(&mut self.config).diagnostics.attention_candidate = candidate;
    }

    #[cfg(test)]
    pub(crate) fn set_force_native_paged_attention(&mut self, force: bool) {
        Arc::make_mut(&mut self.config).cache.force_native_paged_attention = force;
    }

    pub fn new_gpu() -> Result<Self> {
        Self::new(mirtal::Device::gpu(0), Arc::default(), Arc::default())
    }

    #[cfg(test)]
    pub(crate) fn new_gpu_with_config(config: Arc<crate::MetalConfig>) -> Result<Self> {
        Self::new(mirtal::Device::gpu(0), config, Arc::default())
    }

    pub(crate) fn new_gpu_with_config_and_pool(
        config: Arc<crate::MetalConfig>,
        paged_arenas: Arc<PagedArenaPool>,
    ) -> Result<Self> {
        Self::new(mirtal::Device::gpu(0), config, paged_arenas)
    }

    pub fn new_cpu() -> Result<Self> {
        Self::new(mirtal::Device::cpu(0), Arc::default(), Arc::default())
    }

    fn new(
        device: mirtal::Device,
        config: Arc<crate::MetalConfig>,
        paged_arenas: Arc<PagedArenaPool>,
    ) -> Result<Self> {
        let native = device.new_stream()?;
        Ok(Self {
            #[cfg(test)]
            history_budget: Arc::default(),
            compiled: CompiledGraphs::new(&native)?,
            kernels: Kernels::new()?,
            tuner: Mutex::new(MetalTuner::new(config.tuning.clone())),
            native,
            config,
            graph_dumped: AtomicBool::new(false),
            paged_arenas,
        })
    }
}
