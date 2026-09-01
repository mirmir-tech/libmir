use mircuda::{DeviceBuffer, PinnedBuffer, Stream};

use crate::{CudaBackend, Result};

#[derive(Debug)]
pub(super) struct WindowedMetadata {
    pub host: Vec<u32>,
    staging: PinnedBuffer<u32>,
    pub device: DeviceBuffer<u32>,
}

impl WindowedMetadata {
    pub(super) fn new(backend: &CudaBackend, len: usize, fill: u32) -> Result<Self> {
        Ok(Self {
            host: vec![fill; len],
            staging: backend.inner.context.allocate_pinned(len)?,
            device: backend.inner.pool.allocate(&backend.inner.stream, len)?,
        })
    }

    pub(super) fn upload(&mut self, stream: &Stream) -> Result<()> {
        self.staging.copy_from_slice(&self.host)?;
        Ok(stream.copy_to_device(&mut self.staging, &mut self.device)?)
    }
}
