use super::super::{CudaSharedRoutedLayerState, CudaSharedRoutedModelSession};
use crate::{
    CudaAffineGatedFullAttentionState, CudaGatedDeltaState, CudaMamba2State, DecodeAttentionBf16,
    Error, Result,
};

/// Every session's state of layer `index`, which must be of the kind
/// `select` accepts.
fn layer_states<'a, T: ?Sized>(
    sessions: &'a mut [&mut CudaSharedRoutedModelSession],
    index: usize,
    select: fn(&mut CudaSharedRoutedLayerState) -> Option<&mut T>,
) -> Result<Vec<&'a mut T>> {
    sessions
        .iter_mut()
        .map(|session| {
            select(&mut session.states[index])
                .ok_or(Error::InvalidDecoderKernel("packed layer state kind mismatch"))
        })
        .collect()
}

pub(super) fn linear_states<'a>(
    sessions: &'a mut [&mut CudaSharedRoutedModelSession],
    index: usize,
) -> Result<Vec<&'a mut CudaGatedDeltaState>> {
    layer_states(sessions, index, |state| match state {
        CudaSharedRoutedLayerState::Linear(state) => Some(state),
        _ => None,
    })
}

pub(super) fn full_states<'a>(
    sessions: &'a mut [&mut CudaSharedRoutedModelSession],
    index: usize,
) -> Result<Vec<&'a mut CudaAffineGatedFullAttentionState>> {
    layer_states(sessions, index, |state| match state {
        CudaSharedRoutedLayerState::Full(state) => Some(state.as_mut()),
        _ => None,
    })
}

pub(super) fn mamba_states<'a>(
    sessions: &'a mut [&mut CudaSharedRoutedModelSession],
    index: usize,
) -> Result<Vec<&'a mut CudaMamba2State>> {
    layer_states(sessions, index, |state| match state {
        CudaSharedRoutedLayerState::Mamba2(state) => Some(state),
        _ => None,
    })
}

pub(super) fn attention_states<'a>(
    sessions: &'a mut [&mut CudaSharedRoutedModelSession],
    index: usize,
) -> Result<Vec<&'a mut DecodeAttentionBf16>> {
    layer_states(sessions, index, |state| match state {
        CudaSharedRoutedLayerState::Attention(state) => Some(state.as_mut()),
        _ => None,
    })
}
