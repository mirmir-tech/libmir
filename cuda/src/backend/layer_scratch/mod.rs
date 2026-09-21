use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard, Weak},
};

use mircuda::{DeviceBuffer, bf16};

use crate::{CudaBackend, Error, Result};

#[cfg(all(test, target_os = "linux"))]
mod tests;

#[derive(Debug)]
pub(super) struct LayerScratch {
    pub(super) normalized: DeviceBuffer<bf16>,
    pub(super) attention: DeviceBuffer<bf16>,
    pub(super) residual: DeviceBuffer<bf16>,
    pub(super) moe: DeviceBuffer<bf16>,
}

#[derive(Debug)]
pub(super) struct SharedLayerScratch {
    buffers: Arc<Mutex<LayerScratch>>,
    elements: usize,
}

impl SharedLayerScratch {
    pub(super) const fn elements(&self) -> usize {
        self.elements
    }

    /// Lock for the entire layer submission. All users enqueue on one explicit
    /// runtime stream; its ordering also protects reuse during graph replay.
    pub(super) fn lock(&self) -> Result<MutexGuard<'_, LayerScratch>> {
        let Ok(guard) = self.buffers.lock() else {
            return Err(Error::InvalidExecutionPlan("layer scratch lock is poisoned"));
        };
        Ok(guard)
    }
}

/// Weak entries keep no allocation alive after the last execution or graph.
/// Auxiliary runtimes own a separate pool, so streams never alias scratch.
#[derive(Debug, Default)]
pub(super) struct LayerScratchPool {
    entries: Mutex<HashMap<usize, Weak<Mutex<LayerScratch>>>>,
}

impl LayerScratchPool {
    pub(super) fn acquire(
        &self,
        backend: &CudaBackend,
        tokens: usize,
        hidden: usize,
    ) -> Result<SharedLayerScratch> {
        let elements = tokens
            .checked_mul(hidden)
            .ok_or(Error::InvalidExecutionPlan("layer scratch size overflow"))?;
        let Ok(mut entries) = self.entries.lock() else {
            return Err(Error::InvalidExecutionPlan("layer scratch pool lock is poisoned"));
        };
        if let Some(buffers) = entries.get(&elements).and_then(Weak::upgrade) {
            return Ok(SharedLayerScratch { buffers, elements });
        }
        entries.retain(|_, entry| entry.strong_count() > 0);
        let allocate = || backend.inner.pool.allocate(&backend.inner.stream, elements);
        let buffers = Arc::new(Mutex::new(LayerScratch {
            normalized: allocate()?,
            attention: allocate()?,
            residual: allocate()?,
            moe: allocate()?,
        }));
        entries.insert(elements, Arc::downgrade(&buffers));
        drop(entries);
        Ok(SharedLayerScratch { buffers, elements })
    }
}
