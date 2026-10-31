use std::ops::Range;

use mircuda::{DeviceBuffer, bf16};

use super::CudaMamba2State;
use crate::{Result, backend::recurrent::RecurrentDestination};

impl CudaMamba2State {
    /// Identity and revision; changes whenever the state advances or resets.
    pub(super) const fn stamp(&self) -> (u64, u64) {
        (self.identity, self.revision)
    }

    pub(super) fn resident_in(&self, owner: u64, row: usize) -> bool {
        self.resident.resident_in(owner, row)
    }

    pub(super) fn state_source(&self) -> (&DeviceBuffer<f32>, Range<usize>) {
        self.resident.state_source(&self.state)
    }

    pub(super) fn history_source(&self) -> (&DeviceBuffer<bf16>, Range<usize>) {
        self.resident.history_source(&self.history)
    }

    pub(super) fn bind_resident(
        &mut self,
        owner: u64,
        row: usize,
        batch: (DeviceBuffer<f32>, DeviceBuffer<bf16>),
    ) -> RecurrentDestination {
        self.resident.bind(owner, row, batch, (&self.state, &self.history))
    }

    pub(super) fn materialize(&mut self) -> Result<()> {
        let stream = self.backend.inner.stream.clone();
        self.resident.materialize(&stream, &mut self.state, &mut self.history)
    }
}
