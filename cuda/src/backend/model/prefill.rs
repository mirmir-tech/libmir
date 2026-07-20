use mircuda::{DeviceBuffer, PinnedBuffer, Stream};

use super::CudaModelSessionConfig;
use crate::{CudaBackend, Error, Result};

pub(super) struct PrefillTokenBuffer {
    device: DeviceBuffer<u32>,
    staging: PinnedBuffer<u32>,
    capacity: usize,
}

impl PrefillTokenBuffer {
    pub(super) fn new(backend: &CudaBackend, config: CudaModelSessionConfig) -> Result<Self> {
        let capacity = config.validate()?.prefill_chunk_tokens;
        Ok(Self {
            device: backend.inner.pool.allocate::<u32>(&backend.inner.stream, capacity)?,
            staging: backend.inner.context.allocate_pinned(capacity)?,
            capacity,
        })
    }

    pub(super) fn upload(&mut self, stream: &Stream, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() || tokens.len() > self.capacity {
            return Err(Error::InvalidDecoderKernel("invalid CUDA prefill token chunk"));
        }
        self.staging.write_prefix(tokens)?;
        stream.copy_to_device(&mut self.staging, &mut self.device)?;
        Ok(())
    }

    pub(super) const fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) const fn device(&self) -> &DeviceBuffer<u32> {
        &self.device
    }

    pub(super) fn ensure_capacity(&mut self, backend: &CudaBackend, capacity: usize) -> Result<()> {
        if capacity <= self.capacity {
            return Ok(());
        }
        self.device = backend.inner.pool.allocate(&backend.inner.stream, capacity)?;
        self.staging = backend.inner.context.allocate_pinned(capacity)?;
        self.capacity = capacity;
        Ok(())
    }
}
