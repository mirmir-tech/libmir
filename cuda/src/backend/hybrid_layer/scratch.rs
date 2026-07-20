use mircuda::{DeviceBuffer, bf16};

use crate::{CudaBackend, Error, Result};

#[derive(Debug)]
pub(super) struct HybridLayerScratch {
    pub(super) normalized: DeviceBuffer<bf16>,
    pub(super) attention: DeviceBuffer<bf16>,
    pub(super) residual: DeviceBuffer<bf16>,
    pub(super) moe: DeviceBuffer<bf16>,
}

impl HybridLayerScratch {
    pub(super) fn new(backend: &CudaBackend, tokens: usize, hidden: usize) -> Result<Self> {
        let elements = tokens
            .checked_mul(hidden)
            .ok_or(Error::InvalidDecoderKernel("hybrid layer scratch overflow"))?;
        let allocate = || backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            normalized: allocate()?,
            attention: allocate()?,
            residual: allocate()?,
            moe: allocate()?,
        })
    }
}
