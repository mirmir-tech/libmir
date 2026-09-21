use super::{CudaSharedRoutedLayerState, CudaSharedRoutedModelSession};
use crate::{Error, Result, backend::gated_delta::CudaGatedDeltaCheckpoint};

#[derive(Debug)]
pub struct SharedRoutedCheckpoint {
    linear: Vec<Option<CudaGatedDeltaCheckpoint>>,
    pub(super) position: usize,
    pub(super) position_delta: i32,
    bytes: usize,
}

impl SharedRoutedCheckpoint {
    pub(super) fn capture(
        states: &[CudaSharedRoutedLayerState],
        position: usize,
        position_delta: i32,
    ) -> Result<Self> {
        let mut bytes = 0_usize;
        let linear = states
            .iter()
            .map(|state| match state {
                CudaSharedRoutedLayerState::Linear(state) => {
                    bytes = bytes.saturating_add(state.checkpoint_bytes());
                    state.checkpoint().map(Some)
                },
                CudaSharedRoutedLayerState::Full(_) => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { linear, position, position_delta, bytes })
    }

    /// Collects the checkpoints that linear layers staged inside the last
    /// prefill pass; `None` when no layer staged one.
    pub(super) fn take_staged(
        states: &mut [CudaSharedRoutedLayerState],
        position: usize,
        position_delta: i32,
    ) -> Result<Option<Self>> {
        let mut bytes = 0_usize;
        let mut staged = 0_usize;
        let mut expected = 0_usize;
        let linear = states
            .iter_mut()
            .map(|state| match state {
                CudaSharedRoutedLayerState::Linear(state) => {
                    expected += 1;
                    let checkpoint = state.take_staged_checkpoint();
                    if checkpoint.is_some() {
                        staged += 1;
                        bytes = bytes.saturating_add(state.checkpoint_bytes());
                    }
                    checkpoint
                },
                CudaSharedRoutedLayerState::Full(_) => None,
            })
            .collect::<Vec<_>>();
        if staged == 0 {
            return Ok(None);
        }
        if staged != expected {
            return Err(Error::InvalidExecutionPlan(
                "shared-routed layers staged an incomplete checkpoint",
            ));
        }
        Ok(Some(Self { linear, position, position_delta, bytes }))
    }

    pub(super) fn restore(&self, states: &mut [CudaSharedRoutedLayerState]) -> Result<()> {
        if self.linear.len() != states.len() {
            return Err(Error::InvalidDecoderKernel(
                "shared-routed checkpoint layer count mismatch",
            ));
        }
        for (checkpoint, state) in self.linear.iter().zip(states) {
            match (checkpoint, state) {
                (Some(checkpoint), CudaSharedRoutedLayerState::Linear(state)) => {
                    state.restore(checkpoint)?;
                },
                (None, CudaSharedRoutedLayerState::Full(_)) => {},
                _ => {
                    return Err(Error::InvalidDecoderKernel(
                        "shared-routed checkpoint layer kind mismatch",
                    ));
                },
            }
        }
        Ok(())
    }

    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }
}

impl CudaSharedRoutedModelSession {
    /// Requests a checkpoint `tokens` tokens into the next prefill pass, so a
    /// short prompt tail needs no pass of its own.
    pub(crate) fn arm_checkpoint(&mut self, tokens: usize) {
        for state in &mut self.states {
            if let CudaSharedRoutedLayerState::Linear(state) = state {
                state.arm_checkpoint(tokens);
            }
        }
    }

    /// Takes the checkpoint staged at `position` by the last prefill pass and
    /// clears any request the pass left unanswered.
    pub(crate) fn take_staged_checkpoint(
        &mut self,
        position: usize,
    ) -> Result<Option<SharedRoutedCheckpoint>> {
        let position_delta = self.position_delta();
        SharedRoutedCheckpoint::take_staged(&mut self.states, position, position_delta)
    }
}
