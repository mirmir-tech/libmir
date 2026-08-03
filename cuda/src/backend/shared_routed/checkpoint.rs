use super::CudaSharedRoutedLayerState;
use crate::{Error, Result, backend::gated_delta::CudaGatedDeltaCheckpoint};

#[derive(Debug)]
pub(crate) struct SharedRoutedCheckpoint {
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
