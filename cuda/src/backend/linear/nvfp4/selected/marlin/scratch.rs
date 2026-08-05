use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, bf16};

use crate::{CudaBackend, Error, Result};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct MarlinNvFp4ScratchConfig {
    pub tokens: usize,
    pub top_k: usize,
    pub experts: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

#[derive(Debug)]
pub(in crate::backend) struct MarlinNvFp4Scratch {
    pub(super) sorted: DeviceBuffer<i32>,
    pub(super) expert_ids: DeviceBuffer<i32>,
    pub(super) padded: DeviceBuffer<i32>,
    pub(super) offsets: DeviceBuffer<i32>,
    pub(super) routing: DeviceBuffer<f32>,
    pub(super) gate_up: DeviceBuffer<bf16>,
    pub(super) intermediate: DeviceBuffer<bf16>,
    pub(super) down: DeviceBuffer<bf16>,
    pub(super) temporary: DeviceBuffer<f32>,
    pub(super) locks: DeviceBuffer<i32>,
}

impl CudaBackend {
    pub(super) fn marlin_nvfp4_scratch(
        &self,
        config: MarlinNvFp4ScratchConfig,
    ) -> Result<Arc<Mutex<MarlinNvFp4Scratch>>> {
        let mut cache =
            self.inner.nvfp4_marlin_scratch.lock().map_err(|_| {
                Error::InvalidExecutionPlan("Marlin scratch cache lock is poisoned")
            })?;
        if let Some(scratch) = cache.get(&config).and_then(std::sync::Weak::upgrade) {
            return Ok(scratch);
        }
        let scratch = Arc::new(Mutex::new(MarlinNvFp4Scratch::new(self, config)?));
        cache.insert(config, Arc::downgrade(&scratch));
        drop(cache);
        Ok(scratch)
    }
}

impl MarlinNvFp4Scratch {
    fn new(backend: &CudaBackend, config: MarlinNvFp4ScratchConfig) -> Result<Self> {
        let assignments = product(config.tokens, config.top_k)?;
        let capacity = assignments
            .checked_add(product(config.experts, 7)?)
            .ok_or(Error::InvalidNvFp4("Marlin route capacity overflow"))?;
        let maximum_width = config.hidden.max(product(config.intermediate, 2)?);
        Ok(Self {
            sorted: allocate(backend, capacity)?,
            expert_ids: allocate(backend, capacity.div_ceil(8))?,
            padded: allocate(backend, 1)?,
            offsets: allocate(backend, config.experts)?,
            routing: allocate(backend, assignments)?,
            gate_up: allocate(backend, product(assignments, product(config.intermediate, 2)?)?)?,
            intermediate: allocate(backend, product(assignments, config.intermediate)?)?,
            down: allocate(backend, product(assignments, config.hidden)?)?,
            temporary: allocate(backend, product(capacity, maximum_width)?)?,
            locks: backend.inner.pool.allocate_zeroed(
                &backend.inner.stream,
                usize::try_from(backend.inner.device.multiprocessor_count)? * 4,
            )?,
        })
    }
}

fn allocate<T: mircuda::DeviceElement>(
    backend: &CudaBackend,
    elements: usize,
) -> Result<DeviceBuffer<T>> {
    backend.inner.pool.allocate(&backend.inner.stream, elements).map_err(Into::into)
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidNvFp4("Marlin scratch size overflow"))
}
