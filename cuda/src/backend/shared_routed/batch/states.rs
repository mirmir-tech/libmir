use super::super::{CudaSharedRoutedLayerState, CudaSharedRoutedModelSession};
use crate::{CudaGatedDeltaState, Error, Result};

pub(super) fn linear_states<'a>(
    sessions: &'a mut [&mut CudaSharedRoutedModelSession],
    index: usize,
) -> Result<Vec<&'a mut CudaGatedDeltaState>> {
    sessions
        .iter_mut()
        .map(|session| match &mut session.states[index] {
            CudaSharedRoutedLayerState::Linear(state) => Ok(state),
            CudaSharedRoutedLayerState::Full(_) => {
                Err(Error::InvalidDecoderKernel("packed linear layer state mismatch"))
            },
        })
        .collect()
}

pub(super) fn full_states<'a>(
    sessions: &'a mut [&mut CudaSharedRoutedModelSession],
    index: usize,
) -> Result<Vec<&'a mut crate::CudaAffineGatedFullAttentionState>> {
    sessions
        .iter_mut()
        .map(|session| match &mut session.states[index] {
            CudaSharedRoutedLayerState::Full(state) => Ok(state.as_mut()),
            CudaSharedRoutedLayerState::Linear(_) => {
                Err(Error::InvalidDecoderKernel("packed full layer state mismatch"))
            },
        })
        .collect()
}
