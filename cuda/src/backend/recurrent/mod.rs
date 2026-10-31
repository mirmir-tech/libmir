//! Residency of one sequence's recurrent mixer state (FP32 state and BF16
//! convolution history) in a row of a packed decode batch. While resident,
//! the batch row holds the authoritative values: the sequence copies them
//! back before running alone, and the batch preserves them into the
//! sequence's own buffers before another sequence overwrites the row.

use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use mircuda::{DeviceBuffer, Stream, bf16};

use crate::Result;

#[derive(Debug)]
struct RecurrentResidency {
    owner: u64,
    row: usize,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    valid: Arc<AtomicBool>,
}

impl Drop for RecurrentResidency {
    fn drop(&mut self) {
        self.valid.store(false, Ordering::Release);
    }
}

/// Where a sequence's recurrent state currently lives.
#[derive(Debug, Default)]
pub(in crate::backend) struct ResidentRow(Option<Box<RecurrentResidency>>);

/// A batch row's obligation to write a resident sequence back to its own
/// buffers before the row is reused.
#[derive(Debug)]
pub(in crate::backend) struct RecurrentDestination {
    row: usize,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    valid: Arc<AtomicBool>,
}

impl RecurrentDestination {
    pub(in crate::backend) fn preserve(
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

impl ResidentRow {
    fn live(&self) -> Option<&RecurrentResidency> {
        self.0.as_deref().filter(|resident| resident.valid.load(Ordering::Acquire))
    }

    pub(in crate::backend) fn resident_in(&self, owner: u64, row: usize) -> bool {
        self.live()
            .is_some_and(|resident| resident.owner == owner && resident.row == row)
    }

    /// The buffer and range holding the current state of `own`'s length.
    pub(in crate::backend) fn state_source<'a>(
        &'a self,
        own: &'a DeviceBuffer<f32>,
    ) -> (&'a DeviceBuffer<f32>, Range<usize>) {
        self.live().map_or((own, 0..own.len()), |resident| {
            let start = resident.row * own.len();
            (&resident.state, start..start + own.len())
        })
    }

    /// The buffer and range holding the current convolution history.
    pub(in crate::backend) fn history_source<'a>(
        &'a self,
        own: &'a DeviceBuffer<bf16>,
    ) -> (&'a DeviceBuffer<bf16>, Range<usize>) {
        self.live().map_or((own, 0..own.len()), |resident| {
            let start = resident.row * own.len();
            (&resident.history, start..start + own.len())
        })
    }

    /// Marks the sequence resident in `row` of the batch buffers.
    pub(in crate::backend) fn bind(
        &mut self,
        owner: u64,
        row: usize,
        batch: (DeviceBuffer<f32>, DeviceBuffer<bf16>),
        own: (&DeviceBuffer<f32>, &DeviceBuffer<bf16>),
    ) -> RecurrentDestination {
        if !self.resident_in(owner, row) {
            self.0 = None;
        }
        let resident = self.0.get_or_insert_with(|| {
            Box::new(RecurrentResidency {
                owner,
                row,
                state: batch.0,
                history: batch.1,
                valid: Arc::new(AtomicBool::new(true)),
            })
        });
        RecurrentDestination {
            row,
            state: own.0.clone(),
            history: own.1.clone(),
            valid: Arc::clone(&resident.valid),
        }
    }

    /// Copies a resident state back into the sequence's own buffers and ends
    /// the residency.
    pub(in crate::backend) fn materialize(
        &mut self,
        stream: &Stream,
        state: &mut DeviceBuffer<f32>,
        history: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let Some(resident) = self.0.take() else {
            return Ok(());
        };
        if !resident.valid.load(Ordering::Acquire) {
            return Ok(());
        }
        let (state_len, history_len) = (state.len(), history.len());
        let state_start = resident.row * state_len;
        let history_start = resident.row * history_len;
        stream.copy_device_range(
            &resident.state,
            state_start..state_start + state_len,
            state,
            0,
        )?;
        stream.copy_device_range(
            &resident.history,
            history_start..history_start + history_len,
            history,
            0,
        )?;
        Ok(())
    }

    pub(in crate::backend) fn clear(&mut self) {
        self.0 = None;
    }
}
