//! Packed decode of several Mamba-2 sequences. Each sequence's state is
//! copied into a contiguous row once and stays resident there across decode
//! steps; the device work of a step needs no allocation or host
//! synchronization and can be captured in a CUDA graph.

use std::sync::atomic::{AtomicU64, Ordering};

use mircuda::{DeviceBuffer, bf16};

use super::CudaMamba2State;
use crate::{
    CudaBackend, Error, Result,
    backend::recurrent::RecurrentDestination,
    kernels::{Mamba2DecodeBatch, Mamba2Geometry, Mamba2Kernels, Mamba2Parameters},
};

static NEXT_BATCH_IDENTITY: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct CudaMamba2BatchState {
    backend: CudaBackend,
    kernels: Mamba2Kernels,
    capacity: usize,
    rows: usize,
    states: DeviceBuffer<f32>,
    histories: DeviceBuffer<bf16>,
    convolved: DeviceBuffer<bf16>,
    scanned: DeviceBuffer<bf16>,
    sources: Vec<(u64, u64)>,
    destinations: Vec<RecurrentDestination>,
    identity: u64,
}

impl CudaBackend {
    /// Packed decode rows for up to `capacity` sequences of one layer.
    pub fn prepare_mamba2_batch_state(
        &self,
        geometry: Mamba2Geometry,
        capacity: usize,
    ) -> Result<CudaMamba2BatchState> {
        if capacity == 0 {
            return Err(Error::InvalidDecoderKernel("Mamba-2 packed state has no rows"));
        }
        let rows = |elements: usize| {
            elements
                .checked_mul(capacity)
                .ok_or(Error::InvalidDecoderKernel("Mamba-2 packed state overflow"))
        };
        let (stream, pool) = (&self.inner.stream, &self.inner.pool);
        Ok(CudaMamba2BatchState {
            backend: self.clone(),
            kernels: Mamba2Kernels::compile(&self.inner.compiler, geometry)?,
            capacity,
            rows: 0,
            states: pool.allocate(stream, rows(geometry.state_elements()?)?)?,
            histories: pool.allocate(stream, rows(geometry.history_elements()?)?)?,
            convolved: pool.allocate(stream, rows(geometry.convolution_channels()?)?)?,
            scanned: pool.allocate(stream, rows(geometry.inner()?)?)?,
            sources: Vec::new(),
            destinations: Vec::new(),
            identity: NEXT_BATCH_IDENTITY.fetch_add(1, Ordering::Relaxed),
        })
    }
}

impl CudaMamba2BatchState {
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Places `states` in rows `0..states.len()`. Rows already resident in
    /// place are not copied; a displaced resident sequence is first written
    /// back to its own buffers.
    pub fn pack(&mut self, states: &[&mut CudaMamba2State]) -> Result<()> {
        let geometry = self.kernels.geometry();
        if states.is_empty()
            || states.len() > self.capacity
            || states.iter().any(|state| state.geometry() != geometry)
        {
            return Err(Error::InvalidDecoderKernel("Mamba-2 packed state row mismatch"));
        }
        self.rows = states.len();
        if self.sources.len() == states.len()
            && self.sources.iter().zip(states).all(|(source, state)| *source == state.stamp())
            && states
                .iter()
                .enumerate()
                .all(|(row, state)| state.resident_in(self.identity, row))
        {
            return Ok(());
        }
        let stream = &self.backend.inner.stream;
        // Preserve every live previous row before writing any replacement: a
        // permutation aliases these buffers, and omitted sequences may resume.
        for destination in &self.destinations {
            destination.preserve(stream, &self.states, &self.histories)?;
        }
        self.destinations.clear();
        for (row, state) in states.iter().enumerate() {
            if !state.resident_in(self.identity, row) {
                let (source, range) = state.state_source();
                let length = range.len();
                stream.copy_device_range(source, range, &mut self.states, row * length)?;
                let (source, range) = state.history_source();
                let length = range.len();
                stream.copy_device_range(source, range, &mut self.histories, row * length)?;
            }
        }
        self.sources = states.iter().map(|state| state.stamp()).collect();
        Ok(())
    }

    /// Advances every packed row by one token. `projection` holds one fused
    /// input-projection row per packed sequence; `output` receives one gated,
    /// normalized row per sequence.
    pub fn decode(
        &mut self,
        projection: &DeviceBuffer<bf16>,
        parameters: Mamba2Parameters<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let geometry = self.kernels.geometry();
        let rows = self.rows;
        let range = |elements: usize| rows * elements;
        let mut states = self.states.slice(0..range(geometry.state_elements()?))?;
        let mut histories = self.histories.slice(0..range(geometry.history_elements()?))?;
        let mut convolved = self.convolved.slice(0..range(geometry.convolution_channels()?))?;
        let mut scanned = self.scanned.slice(0..range(geometry.inner()?))?;
        self.kernels.execute_decode_batch(
            &self.backend.inner.stream,
            &mut Mamba2DecodeBatch {
                rows,
                projection,
                parameters,
                states: &mut states,
                histories: &mut histories,
                convolved: &mut convolved,
                scanned: &mut scanned,
                output,
            },
        )
    }

    /// Records that `states` advanced by one token and now live in their rows.
    pub fn commit(&mut self, states: &mut [&mut CudaMamba2State]) -> Result<()> {
        if states.len() != self.rows {
            return Err(Error::InvalidDecoderKernel("Mamba-2 packed commit row mismatch"));
        }
        self.destinations.clear();
        for (row, state) in states.iter_mut().enumerate() {
            state.advance(1)?;
            self.destinations.push(state.bind_resident(
                self.identity,
                row,
                (self.states.clone(), self.histories.clone()),
            ));
            self.sources[row] = state.stamp();
        }
        Ok(())
    }
}
