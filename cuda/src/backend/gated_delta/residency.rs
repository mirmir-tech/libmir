use std::ops::Range;

use mircuda::{DeviceBuffer, bf16};

use super::CudaGatedDeltaState;
use crate::{Result, backend::recurrent::RecurrentDestination};

impl CudaGatedDeltaState {
    pub(super) fn resident_in(&self, owner: u64, row: usize) -> bool {
        self.resident.resident_in(owner, row)
    }

    pub(super) fn state_source(&self) -> (&DeviceBuffer<f32>, Range<usize>) {
        self.resident.state_source(&self.state)
    }

    pub(super) fn history_source(&self) -> (&DeviceBuffer<bf16>, Range<usize>) {
        self.resident.history_source(&self.convolution)
    }

    pub(super) fn bind_resident(
        &mut self,
        owner: u64,
        row: usize,
        state: DeviceBuffer<f32>,
        history: DeviceBuffer<bf16>,
    ) -> RecurrentDestination {
        self.resident
            .bind(owner, row, (state, history), (&self.state, &self.convolution))
    }

    pub(super) fn materialize(&mut self) -> Result<()> {
        let stream = self.backend.inner.stream.clone();
        self.resident.materialize(&stream, &mut self.state, &mut self.convolution)
    }

    pub(super) fn clear_residency(&mut self) {
        self.resident.clear();
    }
}
