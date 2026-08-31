use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, MarlinNvFp4ThreadConfig, bf16};

use crate::{CudaBackend, Error, Result};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct MarlinNvFp4ScratchConfig {
    pub tokens: usize,
    pub top_k: usize,
    pub experts: usize,
    pub route_block: MarlinRouteBlock,
    pub hidden: usize,
    pub intermediate: usize,
    pub padded_hidden: usize,
    pub padded_intermediate: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) enum MarlinRouteBlock {
    Eight,
    SixtyFour,
}

impl MarlinRouteBlock {
    const fn size(self) -> usize {
        match self {
            Self::Eight => 8,
            Self::SixtyFour => 64,
        }
    }
}

impl From<MarlinNvFp4ThreadConfig> for MarlinRouteBlock {
    fn from(config: MarlinNvFp4ThreadConfig) -> Self {
        match config {
            MarlinNvFp4ThreadConfig::N128K128
            | MarlinNvFp4ThreadConfig::N128K64
            | MarlinNvFp4ThreadConfig::N64K128 => Self::Eight,
            MarlinNvFp4ThreadConfig::M64N256K64
            | MarlinNvFp4ThreadConfig::M64N128K64
            | MarlinNvFp4ThreadConfig::M64N64K128 => Self::SixtyFour,
        }
    }
}

#[derive(Debug)]
pub(in crate::backend) struct MarlinNvFp4Scratch {
    pub(in crate::backend) sorted: DeviceBuffer<i32>,
    pub(in crate::backend) expert_ids: DeviceBuffer<i32>,
    pub(in crate::backend) padded: DeviceBuffer<i32>,
    pub(in crate::backend) offsets: DeviceBuffer<i32>,
    pub(in crate::backend) routing: DeviceBuffer<f32>,
    pub(in crate::backend) padded_input: DeviceBuffer<bf16>,
    pub(in crate::backend) gate_up: DeviceBuffer<bf16>,
    pub(in crate::backend) intermediate: DeviceBuffer<bf16>,
    pub(in crate::backend) padded_intermediate: DeviceBuffer<bf16>,
    pub(in crate::backend) padded_down: DeviceBuffer<bf16>,
    pub(in crate::backend) down: DeviceBuffer<bf16>,
    pub(in crate::backend) temporary: DeviceBuffer<f32>,
    pub(in crate::backend) locks: DeviceBuffer<i32>,
}

impl CudaBackend {
    pub(in crate::backend) fn marlin_nvfp4_scratch(
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
        let route_block_size = config.route_block.size();
        let capacity = assignments
            .checked_add(product(config.experts, route_block_size - 1)?)
            .ok_or(Error::InvalidNvFp4("Marlin route capacity overflow"))?;
        if config.padded_hidden < config.hidden || config.padded_intermediate < config.intermediate
        {
            return Err(Error::InvalidNvFp4("invalid Marlin padded feature layout"));
        }
        let padded_gate_up = product(config.padded_intermediate, 2)?;
        let maximum_width = config.padded_hidden.max(padded_gate_up);
        Ok(Self {
            sorted: allocate(backend, capacity)?,
            expert_ids: allocate(backend, capacity.div_ceil(route_block_size))?,
            padded: allocate(backend, 1)?,
            offsets: allocate(backend, config.experts)?,
            routing: allocate(backend, assignments)?,
            padded_input: allocate(backend, product(config.tokens, config.padded_hidden)?)?,
            gate_up: allocate(backend, product(assignments, padded_gate_up)?)?,
            intermediate: allocate(backend, product(assignments, config.intermediate)?)?,
            padded_intermediate: allocate(
                backend,
                product(assignments, config.padded_intermediate)?,
            )?,
            padded_down: allocate(backend, product(assignments, config.padded_hidden)?)?,
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
