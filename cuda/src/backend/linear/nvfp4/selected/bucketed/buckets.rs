use mircuda::DeviceBuffer;

use super::CudaBackend;
use crate::Result;

#[derive(Debug)]
pub(in crate::backend::linear::nvfp4::selected) struct ExpertBuckets {
    pub(super) counts: DeviceBuffer<u32>,
    pub(super) offsets: DeviceBuffer<u32>,
    pub(super) scale_offsets: DeviceBuffer<u32>,
    pub(super) order: DeviceBuffer<u32>,
    pub(super) positions: DeviceBuffer<u32>,
    pub(super) indices: DeviceBuffer<u32>,
}

impl ExpertBuckets {
    pub(super) fn new(backend: &CudaBackend, assignments: usize, experts: usize) -> Result<Self> {
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            counts: allocate(experts)?,
            offsets: allocate(experts)?,
            scale_offsets: allocate(experts)?,
            order: allocate(assignments)?,
            positions: allocate(assignments)?,
            indices: allocate(experts)?,
        })
    }
}
