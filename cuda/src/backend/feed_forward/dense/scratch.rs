use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

use mircuda::{DeviceBuffer, bf16};

use crate::{CudaBackend, Error, Result};

#[derive(Debug)]
pub(in crate::backend) struct DenseScratch {
    pub(super) gate_output: DeviceBuffer<bf16>,
    pub(super) up_output: DeviceBuffer<bf16>,
    pub(super) activated: DeviceBuffer<bf16>,
}

/// Scratch belongs to one runtime stream, independently of layer weights.
/// Weak entries let plan eviction release buffers before constructing a new
/// plan.
#[derive(Debug, Default)]
pub(in crate::backend) struct DenseScratchPool {
    entries: Mutex<HashMap<usize, Weak<Mutex<DenseScratch>>>>,
}

impl DenseScratchPool {
    pub(in crate::backend) fn acquire(
        &self,
        backend: &CudaBackend,
        elements: usize,
    ) -> Result<Arc<Mutex<DenseScratch>>> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("dense scratch lock is poisoned"))?;
        if let Some(scratch) = entries.get(&elements).and_then(Weak::upgrade) {
            return Ok(scratch);
        }
        entries.retain(|_, value| value.strong_count() > 0);
        let allocate = || backend.inner.pool.allocate(&backend.inner.stream, elements);
        let scratch = Arc::new(Mutex::new(DenseScratch {
            gate_output: allocate()?,
            up_output: allocate()?,
            activated: allocate()?,
        }));
        entries.insert(elements, Arc::downgrade(&scratch));
        drop(entries);
        Ok(scratch)
    }
}
