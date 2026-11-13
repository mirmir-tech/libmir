//! The session's speculative state: device drafts and Mamba-2 rollback.

use super::{super::CudaSharedRoutedLayerState, CudaSharedRoutedModelSession};
use crate::{Error, Result};

impl CudaSharedRoutedModelSession {
    /// Drops the session's drafts after a step without the prediction
    /// module, whose K/V it left behind.
    pub fn forget_drafts(&mut self) {
        self.drafted = 0;
    }

    /// Drafts waiting on the device for the next verification.
    #[must_use]
    pub const fn pending_drafts(&self) -> usize {
        self.drafted
    }

    /// The device draft buffer, holding at least `count` drafts.
    pub(in crate::backend::shared::routed) fn draft_buffer(
        &mut self,
        count: usize,
    ) -> Result<&mut mircuda::DeviceBuffer<u32>> {
        if self.drafts.as_ref().is_none_or(|drafts| drafts.len() < count) {
            let backend = &self.template.backend;
            self.drafts = Some(backend.inner.pool.allocate(&backend.inner.stream, count.max(1))?);
        }
        self.drafts
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("draft buffer is missing"))
    }

    /// Arms Mamba-2 snapshots after each of the first `positions` tokens of
    /// the next pass.
    pub(in crate::backend::shared::routed) fn arm_rollback(
        &mut self,
        positions: usize,
    ) -> Result<()> {
        for state in &mut self.states {
            match state {
                CudaSharedRoutedLayerState::Mamba2(state) => state.arm_rollback(positions)?,
                CudaSharedRoutedLayerState::Linear(_) => {
                    return Err(Error::UnsupportedDecoderLayer(
                        "speculative decoding cannot roll back Gated Delta state".into(),
                    ));
                },
                CudaSharedRoutedLayerState::Full(_) | CudaSharedRoutedLayerState::Attention(_) => {
                },
            }
        }
        Ok(())
    }

    /// Keeps the state after `kept` tokens of the last armed pass, which
    /// started at `start`.
    pub(in crate::backend::shared::routed) fn roll_back(
        &mut self,
        start: usize,
        kept: usize,
    ) -> Result<()> {
        for state in &mut self.states {
            if let CudaSharedRoutedLayerState::Mamba2(state) = state {
                state.roll_back(kept)?;
            }
        }
        self.position = start + kept;
        Ok(())
    }
}
