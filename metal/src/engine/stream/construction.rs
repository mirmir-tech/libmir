use std::sync::{Arc, Mutex, atomic::AtomicBool};

use super::Stream;
use crate::engine::{
    PagedArenaPool, Result, compiled::CompiledGraphs, gate_up_tuning::MetalTuner, kernels::Kernels,
};

impl Stream {
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
