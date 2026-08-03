use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, bf16};

use super::moe::ExpertBuckets;
use crate::{CudaBackend, Error, Result, kernels::scale_elements};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct BucketedNvFp4ScratchConfig {
    pub tokens: usize,
    pub selected: usize,
    pub experts: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

#[derive(Debug)]
pub(in crate::backend) struct ProjectionScratch {
    pub packed: DeviceBuffer<u8>,
    pub scales: DeviceBuffer<u8>,
}

#[derive(Debug)]
pub(in crate::backend) struct BucketedNvFp4Scratch {
    pub(super) buckets: ExpertBuckets,
    pub(super) gate: ProjectionScratch,
    pub(super) up: ProjectionScratch,
    pub(super) down: ProjectionScratch,
    pub(super) gate_output: DeviceBuffer<bf16>,
    pub(super) up_output: DeviceBuffer<bf16>,
    pub(super) intermediate: DeviceBuffer<bf16>,
    pub(super) down_output: DeviceBuffer<bf16>,
}

impl CudaBackend {
    pub(super) fn bucketed_nvfp4_scratch(
        &self,
        config: BucketedNvFp4ScratchConfig,
    ) -> Result<Arc<Mutex<BucketedNvFp4Scratch>>> {
        let mut cache = self.inner.nvfp4_bucket_scratch.lock().map_err(|_| {
            Error::InvalidExecutionPlan("NVFP4 bucket scratch cache lock is poisoned")
        })?;
        if let Some(scratch) = cache.get(&config).and_then(std::sync::Weak::upgrade) {
            return Ok(scratch);
        }
        let scratch = Arc::new(Mutex::new(BucketedNvFp4Scratch::new(self, config)?));
        cache.insert(config, Arc::downgrade(&scratch));
        drop(cache);
        Ok(scratch)
    }
}

impl BucketedNvFp4Scratch {
    fn new(backend: &CudaBackend, config: BucketedNvFp4ScratchConfig) -> Result<Self> {
        let assignments = config
            .tokens
            .checked_mul(config.selected)
            .ok_or(Error::InvalidNvFp4("bucketed assignment count overflow"))?;
        let output = |features| product(assignments, features);
        Ok(Self {
            buckets: ExpertBuckets::new(backend, assignments, config.experts)?,
            gate: ProjectionScratch::new(backend, assignments, config.experts, config.hidden)?,
            up: ProjectionScratch::new(backend, assignments, config.experts, config.hidden)?,
            down: ProjectionScratch::new(
                backend,
                assignments,
                config.experts,
                config.intermediate,
            )?,
            gate_output: allocate(backend, output(config.intermediate)?)?,
            up_output: allocate(backend, output(config.intermediate)?)?,
            intermediate: allocate(backend, output(config.intermediate)?)?,
            down_output: allocate(backend, output(config.hidden)?)?,
        })
    }
}

impl ProjectionScratch {
    fn new(
        backend: &CudaBackend,
        assignments: usize,
        experts: usize,
        columns: usize,
    ) -> Result<Self> {
        let packed = product(assignments, columns / 2)?;
        let padded_rows = assignments
            .checked_add(product(experts, 127)?)
            .ok_or(Error::InvalidNvFp4("bucketed scale capacity overflow"))?
            / 128
            * 128;
        let scales = scale_elements(padded_rows, columns)?;
        Ok(Self {
            packed: allocate(backend, packed)?,
            scales: allocate(backend, scales)?,
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
        .ok_or(Error::InvalidNvFp4("bucketed scratch size overflow"))
}
