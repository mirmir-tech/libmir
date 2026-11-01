//! Per-position snapshots of a verification pass. Speculative decoding
//! advances the state over the fed token and every draft, then keeps the
//! state after the accepted prefix.

use mircuda::{DeviceBuffer, bf16};

use super::{CudaMamba2Checkpoint, CudaMamba2State, Mamba2ScanPath};
use crate::{Error, Result, kernels::Mamba2Parameters};

/// Snapshot buffers reused across steps; `slots[j]` holds the state after
/// `j + 1` tokens of the last armed pass.
#[derive(Debug, Default)]
pub(super) struct Rollback {
    armed: usize,
    slots: Vec<CudaMamba2Checkpoint>,
}

impl CudaMamba2State {
    /// Snapshots the state after each of the first `positions` tokens of the
    /// next pass.
    pub(crate) fn arm_rollback(&mut self, positions: usize) -> Result<()> {
        let stream = &self.backend.inner.stream;
        let pool = &self.backend.inner.pool;
        while self.rollback.slots.len() < positions {
            self.rollback.slots.push(CudaMamba2Checkpoint {
                state: pool.allocate(stream, self.state.len())?,
                history: pool.allocate(stream, self.history.len())?,
                offset: 0,
            });
        }
        self.rollback.armed = positions;
        Ok(())
    }

    pub(super) const fn rollback_armed(&self) -> usize {
        self.rollback.armed
    }

    /// Keeps the state after `kept` tokens of the last armed pass; keeping
    /// every token keeps the current state.
    pub(crate) fn roll_back(&mut self, kept: usize) -> Result<()> {
        let armed = std::mem::take(&mut self.rollback.armed);
        let Some(index) = kept.checked_sub(1).filter(|index| *index < armed) else {
            return Ok(());
        };
        let slot = self
            .rollback
            .slots
            .get(index)
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 rollback slot is missing"))?;
        let stream = &self.backend.inner.stream;
        self.resident.clear();
        stream.copy_device_range(&slot.state, 0..slot.state.len(), &mut self.state, 0)?;
        stream.copy_device_range(&slot.history, 0..slot.history.len(), &mut self.history, 0)?;
        self.offset = slot.offset;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    /// Advances token by token over the armed positions, snapshotting after
    /// each, then over the rest of the chunk.
    pub(super) fn advance_snapshotted(
        &mut self,
        path: Mamba2ScanPath,
        tokens: usize,
        projection: &DeviceBuffer<bf16>,
        parameters: Mamba2Parameters<'_>,
        workspace: &mut super::Mamba2Workspace,
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        let geometry = self.geometry();
        let (width, inner) = (geometry.projection_width()?, geometry.inner()?);
        let snapshots = self.rollback.armed.min(tokens);
        for token in 0..snapshots {
            let mut row = output.slice(token * inner..(token + 1) * inner)?;
            self.advance_chunk(
                Mamba2ScanPath::Sequential,
                1,
                &projection.slice(token * width..(token + 1) * width)?,
                parameters,
                workspace,
                &mut row,
            )?;
            self.snapshot(token)?;
        }
        if snapshots == tokens {
            return Ok(());
        }
        let rest = tokens - snapshots;
        let mut rest_output = output.slice(snapshots * inner..tokens * inner)?;
        self.advance_chunk(
            Mamba2ScanPath::select(rest, geometry).min_path(path),
            rest,
            &projection.slice(snapshots * width..tokens * width)?,
            parameters,
            workspace,
            &mut rest_output,
        )
    }

    fn snapshot(&mut self, index: usize) -> Result<()> {
        let stream = &self.backend.inner.stream;
        let (state, state_range) = self.resident.state_source(&self.state);
        let (history, history_range) = self.resident.history_source(&self.history);
        let slot = self
            .rollback
            .slots
            .get_mut(index)
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 rollback slot is missing"))?;
        stream.copy_device_range(state, state_range, &mut slot.state, 0)?;
        stream.copy_device_range(history, history_range, &mut slot.history, 0)?;
        slot.offset = self.offset;
        Ok(())
    }
}
