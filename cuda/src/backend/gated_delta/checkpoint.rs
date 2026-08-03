use mircuda::{DeviceBuffer, bf16};

use super::CudaGatedDeltaState;
use crate::Result;

#[derive(Debug)]
pub(crate) struct CudaGatedDeltaCheckpoint {
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
        stream.copy_device_range(&self.state, 0..self.state.len(), &mut state, 0)?;
        stream.copy_device_range(
            &self.convolution,
            0..self.convolution.len(),
            &mut convolution,
            0,
        )?;
        Ok(CudaGatedDeltaCheckpoint { state, convolution, offset: self.offset })
    }

    pub(crate) fn restore(&mut self, checkpoint: &CudaGatedDeltaCheckpoint) -> Result<()> {
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
        Ok(())
    }

    pub(crate) fn checkpoint_bytes(&self) -> usize {
        self.state.bytes().saturating_add(self.convolution.bytes())
    }
}
