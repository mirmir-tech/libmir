use std::ops::Range;

use mircuda::{DeviceBuffer, bf16};

use super::CudaGatedDeltaState;
use crate::Result;

#[derive(Debug)]
pub(super) struct GatedDeltaResidency {
    owner: u64,
    row: usize,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
}

impl CudaGatedDeltaState {
    pub(super) fn resident_in(&self, owner: u64, row: usize) -> bool {
        self.resident
            .as_ref()
            .is_some_and(|resident| resident.owner == owner && resident.row == row)
    }

    pub(super) fn state_source(&self) -> (&DeviceBuffer<f32>, Range<usize>) {
        self.resident.as_ref().map_or_else(
            || (&self.state, 0..self.state.len()),
            |resident| {
                let start = resident.row * self.state.len();
                (&resident.state, start..start + self.state.len())
            },
        )
    }

    pub(super) fn history_source(&self) -> (&DeviceBuffer<bf16>, Range<usize>) {
        self.resident.as_ref().map_or_else(
            || (&self.convolution, 0..self.convolution.len()),
            |resident| {
                let start = resident.row * self.convolution.len();
                (&resident.history, start..start + self.convolution.len())
            },
        )
    }

    pub(super) fn bind_resident(
        &mut self,
        owner: u64,
        row: usize,
        state: DeviceBuffer<f32>,
        history: DeviceBuffer<bf16>,
    ) {
        self.resident = Some(GatedDeltaResidency { owner, row, state, history });
    }

    pub(super) fn materialize(&mut self) -> Result<()> {
        let Some(resident) = self.resident.take() else {
            return Ok(());
        };
        let state_start = resident.row * self.state.len();
        let history_start = resident.row * self.convolution.len();
        let stream = &self.backend.inner.stream;
        stream.copy_device_range(
            &resident.state,
            state_start..state_start + self.state.len(),
            &mut self.state,
            0,
        )?;
        stream.copy_device_range(
            &resident.history,
            history_start..history_start + self.convolution.len(),
            &mut self.convolution,
            0,
        )?;
        Ok(())
    }

    pub(super) fn clear_residency(&mut self) {
        self.resident = None;
    }
}
