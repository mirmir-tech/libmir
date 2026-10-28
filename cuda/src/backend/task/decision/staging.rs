use mircuda::{DeviceBuffer, PinnedBuffer};

use crate::{CudaBackend, Result};

/// Pinned host memory reused across calls, since allocating it costs
/// milliseconds; both buffers only ever grow.
pub struct Staging {
    upload: PinnedBuffer<u32>,
    download: PinnedBuffer<f32>,
}

impl Staging {
    pub fn new(backend: &CudaBackend, tokens: usize) -> Result<Self> {
        let context = &backend.inner.context;
        Ok(Self {
            upload: context.allocate_pinned(tokens.next_power_of_two())?,
            download: context.allocate_pinned(tokens.next_power_of_two())?,
        })
    }

    /// Copies every part into a device buffer of its own with one transfer.
    pub fn upload(
        &mut self,
        backend: &CudaBackend,
        parts: &[&[u32]],
    ) -> Result<Vec<DeviceBuffer<u32>>> {
        let inner = &backend.inner;
        let packed = parts.concat();
        if self.upload.len() < packed.len() {
            self.upload = inner.context.allocate_pinned(packed.len().next_power_of_two())?;
        }
        self.upload.write_prefix(&packed)?;
        let mut staged = inner.pool.allocate(&inner.stream, self.upload.len())?;
        inner.stream.copy_to_device(&mut self.upload, &mut staged)?;
        let mut start = 0;
        parts
            .iter()
            .map(|part| {
                let mut target = inner.pool.allocate(&inner.stream, part.len())?;
                let range = start..start + part.len();
                start = range.end;
                if !range.is_empty() {
                    inner.stream.copy_device_range(&staged, range, &mut target, 0)?;
                }
                Ok(target)
            })
            .collect()
    }

    /// Waits for `source` and returns its values.
    pub fn download(
        &mut self,
        backend: &CudaBackend,
        source: &DeviceBuffer<f32>,
    ) -> Result<Vec<f32>> {
        let inner = &backend.inner;
        if self.download.len() < source.len() {
            self.download = inner.context.allocate_pinned(source.len().next_power_of_two())?;
        }
        let mut staged = inner.pool.allocate(&inner.stream, self.download.len())?;
        if !source.is_empty() {
            inner.stream.copy_device_range(source, 0..source.len(), &mut staged, 0)?;
        }
        inner.stream.copy_to_host(&staged, &mut self.download)?;
        let mut values = self.download.to_vec()?;
        values.truncate(source.len());
        Ok(values)
    }
}
