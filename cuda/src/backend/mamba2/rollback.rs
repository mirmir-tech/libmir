//! Per-position snapshots of a verification pass. Speculative decoding
//! advances the state over the fed token and every draft, then keeps the
//! state after the accepted prefix.

use mircuda::{DeviceBuffer, bf16};

use super::CudaMamba2State;
use crate::{Error, Result, kernels::Mamba2Parameters};

/// Snapshot buffers reused across steps: state `j` of `states` and
/// `histories[j]` hold the state after `j + 1` tokens of the last armed
/// pass, which started at `start`.
#[derive(Debug, Default)]
pub(super) struct Rollback {
    armed: usize,
    start: usize,
    states: Option<DeviceBuffer<f32>>,
    histories: Vec<DeviceBuffer<bf16>>,
}

impl CudaMamba2State {
    /// Snapshots the state after each of the first `positions` tokens of the
    /// next pass.
    pub(crate) fn arm_rollback(&mut self, positions: usize) -> Result<()> {
        let stream = &self.backend.inner.stream;
        let pool = &self.backend.inner.pool;
        let elements = positions
            .checked_mul(self.state.len())
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 snapshot size overflow"))?;
        if self.rollback.states.as_ref().is_none_or(|states| states.len() < elements) {
            self.rollback.states = Some(pool.allocate(stream, elements.max(1))?);
        }
        while self.rollback.histories.len() < positions {
            self.rollback.histories.push(pool.allocate(stream, self.history.len())?);
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
        let states = self
            .rollback
            .states
            .as_ref()
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 rollback states are missing"))?;
        let history = self
            .rollback
            .histories
            .get(index)
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 rollback slot is missing"))?;
        let length = self.state.len();
        let stream = &self.backend.inner.stream;
        self.resident.clear();
        stream.copy_device_range(
            states,
            index * length..(index + 1) * length,
            &mut self.state,
            0,
        )?;
        stream.copy_device_range(history, 0..history.len(), &mut self.history, 0)?;
        self.offset = self.rollback.start + kept;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    /// Advances over the whole chunk in one sequential scan that also writes
    /// the state after each armed position.
    pub(super) fn advance_snapshotted(
        &mut self,
        tokens: usize,
        projection: &DeviceBuffer<bf16>,
        parameters: Mamba2Parameters<'_>,
        workspace: &mut super::Mamba2Workspace,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.rollback.armed >= tokens {
            return Err(Error::InvalidDecoderKernel(
                "Mamba-2 snapshots must leave the chunk's last token",
            ));
        }
        self.materialize()?;
        self.rollback.start = self.offset;
        let Rollback { armed, states, histories, .. } = &mut *self.rollback;
        let states = states
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 rollback states are missing"))?;
        let mut histories = histories.iter_mut().take(*armed).collect::<Vec<_>>();
        let mut launch = crate::kernels::Mamba2Launch {
            tokens,
            projection,
            parameters,
            state: &mut self.state,
            history: &self.history,
            next_history: &mut self.next_history,
            convolved: &mut workspace.convolved,
            scanned: &mut workspace.scanned,
            output,
        };
        let stream = &self.backend.inner.stream;
        self.kernels.execute_snapshotted(stream, &mut launch, states, &mut histories)?;
        std::mem::swap(&mut self.history, &mut self.next_history);
        self.advance(tokens)
    }
}
