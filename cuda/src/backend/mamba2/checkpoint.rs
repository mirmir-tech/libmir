use mircuda::{DeviceBuffer, bf16};

use super::CudaMamba2State;
use crate::Result;

/// Copy of one sequence's Mamba-2 state at a token offset.
#[derive(Debug)]
pub struct CudaMamba2Checkpoint {
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    offset: usize,
}

/// A checkpoint requested inside the next prefill pass.
#[derive(Debug)]
pub(super) enum InlineCheckpoint {
    Armed { tokens: usize },
    Staged(CudaMamba2Checkpoint),
}

impl CudaMamba2State {
    /// Requests a checkpoint after `tokens` tokens of the next prefill pass.
    pub(crate) fn arm_checkpoint(&mut self, tokens: usize) {
        self.inline = Some(InlineCheckpoint::Armed { tokens });
    }

    pub(crate) const fn armed_checkpoint(&self) -> Option<usize> {
        match &self.inline {
            Some(InlineCheckpoint::Armed { tokens }) => Some(*tokens),
            _ => None,
        }
    }

    pub(super) fn stage_checkpoint(&mut self) -> Result<()> {
        self.inline = Some(InlineCheckpoint::Staged(self.checkpoint()?));
        Ok(())
    }

    /// Takes the checkpoint staged by the last pass and clears any request.
    pub(crate) fn take_staged_checkpoint(&mut self) -> Option<CudaMamba2Checkpoint> {
        match self.inline.take() {
            Some(InlineCheckpoint::Staged(checkpoint)) => Some(checkpoint),
            _ => None,
        }
    }

    pub(crate) fn checkpoint_bytes(&self) -> usize {
        self.state.bytes().saturating_add(self.history.bytes())
    }

    pub fn checkpoint(&self) -> Result<CudaMamba2Checkpoint> {
        let stream = &self.backend.inner.stream;
        let pool = &self.backend.inner.pool;
        let mut state = pool.allocate(stream, self.state.len())?;
        let mut history = pool.allocate(stream, self.history.len())?;
        let (source, range) = self.state_source();
        stream.copy_device_range(source, range, &mut state, 0)?;
        let (source, range) = self.history_source();
        stream.copy_device_range(source, range, &mut history, 0)?;
        Ok(CudaMamba2Checkpoint { state, history, offset: self.offset })
    }

    pub fn restore(&mut self, checkpoint: &CudaMamba2Checkpoint) -> Result<()> {
        self.resident.clear();
        self.inline = None;
        let stream = &self.backend.inner.stream;
        stream.copy_device_range(
            &checkpoint.state,
            0..checkpoint.state.len(),
            &mut self.state,
            0,
        )?;
        stream.copy_device_range(
            &checkpoint.history,
            0..checkpoint.history.len(),
            &mut self.history,
            0,
        )?;
        self.offset = checkpoint.offset;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }
}

impl CudaMamba2Checkpoint {
    /// Device bytes retained by this checkpoint.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.state.bytes().saturating_add(self.history.bytes())
    }

    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }
}
