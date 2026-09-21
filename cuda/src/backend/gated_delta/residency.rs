use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use mircuda::{DeviceBuffer, Stream, bf16};

use super::CudaGatedDeltaState;
use crate::Result;

#[derive(Debug)]
pub(super) struct GatedDeltaResidency {
    owner: u64,
    row: usize,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    valid: Arc<AtomicBool>,
}

#[derive(Debug)]
pub(super) struct GatedDeltaDestination {
    row: usize,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    valid: Arc<AtomicBool>,
}

impl GatedDeltaDestination {
    pub(super) fn preserve(
        &self,
        stream: &Stream,
        state: &DeviceBuffer<f32>,
        history: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        if !self.valid.load(Ordering::Acquire) {
            return Ok(());
        }
        let start = self.row * self.state.len();
        stream.copy_device_range(
            state,
            start..start + self.state.len(),
            &mut self.state.clone(),
            0,
        )?;
        let start = self.row * self.history.len();
        stream.copy_device_range(
            history,
            start..start + self.history.len(),
            &mut self.history.clone(),
            0,
        )?;
        self.valid.store(false, Ordering::Release);
        Ok(())
    }
}

impl Drop for GatedDeltaResidency {
    fn drop(&mut self) {
        self.valid.store(false, Ordering::Release);
    }
}

impl CudaGatedDeltaState {
    pub(super) fn resident_in(&self, owner: u64, row: usize) -> bool {
        self.resident.as_ref().is_some_and(|resident| {
            resident.owner == owner && resident.row == row && resident.valid.load(Ordering::Acquire)
        })
    }

    pub(super) fn state_source(&self) -> (&DeviceBuffer<f32>, Range<usize>) {
        self.resident
            .as_ref()
            .filter(|resident| resident.valid.load(Ordering::Acquire))
            .map_or_else(
                || (&self.state, 0..self.state.len()),
                |resident| {
                    let start = resident.row * self.state.len();
                    (&resident.state, start..start + self.state.len())
                },
            )
    }

    pub(super) fn history_source(&self) -> (&DeviceBuffer<bf16>, Range<usize>) {
        self.resident
            .as_ref()
            .filter(|resident| resident.valid.load(Ordering::Acquire))
            .map_or_else(
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
    ) -> GatedDeltaDestination {
        if !self.resident_in(owner, row) {
            self.resident = None;
        }
        let resident = self.resident.get_or_insert_with(|| {
            Box::new(GatedDeltaResidency {
                owner,
                row,
                state,
                history,
                valid: Arc::new(AtomicBool::new(true)),
            })
        });
        GatedDeltaDestination {
            row,
            state: self.state.clone(),
            history: self.convolution.clone(),
            valid: Arc::clone(&resident.valid),
        }
    }

    pub(super) fn materialize(&mut self) -> Result<()> {
        let Some(resident) = self.resident.take() else {
            return Ok(());
        };
        if !resident.valid.load(Ordering::Acquire) {
            return Ok(());
        }
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
