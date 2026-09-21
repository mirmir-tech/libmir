use mircuda::{DeviceBuffer, bf16};

use super::CudaGatedDeltaState;
use crate::{Error, Result};

#[derive(Debug)]
pub struct CudaGatedDeltaCheckpoint {
    state: DeviceBuffer<f32>,
    convolution: DeviceBuffer<bf16>,
    offset: usize,
}

/// A checkpoint requested inside the next prefill pass of one state. The
/// convolution history and the recurrent state reach the checkpoint position
/// at different points of the layer, so they are captured separately.
#[derive(Debug)]
pub(super) enum InlineCheckpoint {
    Armed { tokens: usize },
    Convolved { convolution: DeviceBuffer<bf16> },
    Staged(CudaGatedDeltaCheckpoint),
}

impl CudaGatedDeltaState {
    /// Requests a checkpoint after `tokens` tokens of the next prefill pass.
    pub(crate) fn arm_checkpoint(&mut self, tokens: usize) {
        self.inline = Some(Box::new(InlineCheckpoint::Armed { tokens }));
    }

    pub(crate) fn armed_checkpoint(&self) -> Option<usize> {
        match self.inline.as_deref() {
            Some(InlineCheckpoint::Armed { tokens }) => Some(*tokens),
            _ => None,
        }
    }

    pub(super) fn stage_convolution(&mut self) -> Result<()> {
        let Some(InlineCheckpoint::Armed { .. }) = self.inline.as_deref() else {
            return Err(Error::InvalidExecutionPlan("Gated Delta checkpoint is not armed"));
        };
        let stream = &self.backend.inner.stream;
        let (source, range) = self.history_source();
        let mut convolution = self.backend.inner.pool.allocate(stream, range.len())?;
        stream.copy_device_range(source, range, &mut convolution, 0)?;
        self.inline = Some(Box::new(InlineCheckpoint::Convolved { convolution }));
        Ok(())
    }

    pub(super) fn stage_state(&mut self) -> Result<()> {
        let Some(InlineCheckpoint::Convolved { convolution }) =
            self.inline.take().map(|inline| *inline)
        else {
            return Err(Error::InvalidExecutionPlan("Gated Delta checkpoint has no convolution"));
        };
        let stream = &self.backend.inner.stream;
        let (source, range) = self.state_source();
        let mut state = self.backend.inner.pool.allocate(stream, range.len())?;
        stream.copy_device_range(source, range, &mut state, 0)?;
        self.inline = Some(Box::new(InlineCheckpoint::Staged(CudaGatedDeltaCheckpoint {
            state,
            convolution,
            offset: self.offset,
        })));
        Ok(())
    }

    /// Takes the checkpoint staged by the last pass and clears any request.
    pub(crate) fn take_staged_checkpoint(&mut self) -> Option<CudaGatedDeltaCheckpoint> {
        match self.inline.take().map(|inline| *inline) {
            Some(InlineCheckpoint::Staged(checkpoint)) => Some(checkpoint),
            _ => None,
        }
    }

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
        self.inline = None;
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
