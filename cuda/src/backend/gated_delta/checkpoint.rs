use mircuda::{DeviceBuffer, bf16};

use super::CudaGatedDeltaState;
use crate::Result;

#[derive(Debug)]
pub struct CudaGatedDeltaCheckpoint {
    state: DeviceBuffer<f32>,
    convolution: DeviceBuffer<bf16>,
    offset: usize,
}

impl CudaGatedDeltaState {
    pub(crate) fn checkpoint(&self) -> Result<CudaGatedDeltaCheckpoint> {
        let stream = &self.backend.inner.stream;
        let pool = &self.backend.inner.pool;
        let mut state = pool.allocate(stream, self.state.len())?;
        let mut convolution = pool.allocate(stream, self.convolution.len())?;
        let (source, range) = self.state_source();
        stream.copy_device_range(source, range, &mut state, 0)?;
        let (source, range) = self.history_source();
        stream.copy_device_range(source, range, &mut convolution, 0)?;
        Ok(CudaGatedDeltaCheckpoint { state, convolution, offset: self.offset })
    }

    pub(crate) fn restore(&mut self, checkpoint: &CudaGatedDeltaCheckpoint) -> Result<()> {
        self.clear_residency();
        let stream = &self.backend.inner.stream;
        stream.copy_device_range(
            &checkpoint.state,
            0..checkpoint.state.len(),
            &mut self.state,
            0,
        )?;
        stream.copy_device_range(
            &checkpoint.convolution,
            0..checkpoint.convolution.len(),
            &mut self.convolution,
            0,
        )?;
        self.offset = checkpoint.offset;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    pub(crate) fn checkpoint_bytes(&self) -> usize {
        self.state.bytes().saturating_add(self.convolution.bytes())
    }
}
