use super::{CudaSharedRoutedLayerState, CudaSharedRoutedModelSession};
use crate::{CudaMamba2Checkpoint, Error, Result, backend::gated_delta::CudaGatedDeltaCheckpoint};

/// Saved state of one recurrent layer; attention layers keep their K/V in
/// the shared pages and save nothing here.
#[derive(Debug)]
enum RecurrentCheckpoint {
    Linear(CudaGatedDeltaCheckpoint),
    Mamba2(CudaMamba2Checkpoint),
}

#[derive(Debug)]
pub struct SharedRoutedCheckpoint {
    layers: Vec<Option<RecurrentCheckpoint>>,
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
        let layers = states
            .iter()
            .map(|state| {
                Ok(match state {
                    CudaSharedRoutedLayerState::Linear(state) => {
                        bytes = bytes.saturating_add(state.checkpoint_bytes());
                        Some(RecurrentCheckpoint::Linear(state.checkpoint()?))
                    },
                    CudaSharedRoutedLayerState::Mamba2(state) => {
                        bytes = bytes.saturating_add(state.checkpoint_bytes());
                        Some(RecurrentCheckpoint::Mamba2(state.checkpoint()?))
                    },
                    CudaSharedRoutedLayerState::Full(_)
                    | CudaSharedRoutedLayerState::Attention(_) => None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers, position, position_delta, bytes })
    }

    /// Collects the checkpoints that recurrent layers staged inside the last
    /// prefill pass; `None` when no layer staged one.
    pub(super) fn take_staged(
        states: &mut [CudaSharedRoutedLayerState],
        position: usize,
        position_delta: i32,
    ) -> Result<Option<Self>> {
        let mut bytes = 0_usize;
        let mut staged = 0_usize;
        let mut expected = 0_usize;
        let mut count = |checkpoint: Option<RecurrentCheckpoint>, size: usize| {
            expected += 1;
            if checkpoint.is_some() {
                staged += 1;
                bytes = bytes.saturating_add(size);
            }
            checkpoint
        };
        let layers = states
            .iter_mut()
            .map(|state| match state {
                CudaSharedRoutedLayerState::Linear(state) => count(
                    state.take_staged_checkpoint().map(RecurrentCheckpoint::Linear),
                    state.checkpoint_bytes(),
                ),
                CudaSharedRoutedLayerState::Mamba2(state) => count(
                    state.take_staged_checkpoint().map(RecurrentCheckpoint::Mamba2),
                    state.checkpoint_bytes(),
                ),
                CudaSharedRoutedLayerState::Full(_) | CudaSharedRoutedLayerState::Attention(_) => {
                    None
                },
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
        Ok(Some(Self { layers, position, position_delta, bytes }))
    }

    pub(super) fn restore(&self, states: &mut [CudaSharedRoutedLayerState]) -> Result<()> {
        if self.layers.len() != states.len() {
            return Err(Error::InvalidDecoderKernel(
                "shared-routed checkpoint layer count mismatch",
            ));
        }
        for (checkpoint, state) in self.layers.iter().zip(states) {
            match (checkpoint, state) {
                (
                    Some(RecurrentCheckpoint::Linear(checkpoint)),
                    CudaSharedRoutedLayerState::Linear(state),
                ) => state.restore(checkpoint)?,
                (
                    Some(RecurrentCheckpoint::Mamba2(checkpoint)),
                    CudaSharedRoutedLayerState::Mamba2(state),
                ) => state.restore(checkpoint)?,
                (
                    None,
                    CudaSharedRoutedLayerState::Full(_) | CudaSharedRoutedLayerState::Attention(_),
                ) => {},
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
            match state {
                CudaSharedRoutedLayerState::Linear(state) => state.arm_checkpoint(tokens),
                CudaSharedRoutedLayerState::Mamba2(state) => state.arm_checkpoint(tokens),
                CudaSharedRoutedLayerState::Full(_) | CudaSharedRoutedLayerState::Attention(_) => {
                },
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
