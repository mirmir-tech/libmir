//! One sequence's verification pass at fixed addresses, so it can be
//! captured in a CUDA graph: the sequence's state stays resident here across
//! steps, the scan writes the state after each draft position to its own
//! snapshots, and the step keeps the snapshot after the accepted prefix.

use std::sync::atomic::{AtomicU64, Ordering};

use mircuda::{DeviceBuffer, bf16};

use super::CudaMamba2State;
use crate::{
    CudaBackend, Error, Result,
    backend::recurrent::RecurrentDestination,
    kernels::{Mamba2Geometry, Mamba2Kernels, Mamba2Launch, Mamba2Parameters},
};

static NEXT_VERIFY_IDENTITY: AtomicU64 = AtomicU64::new(1 << 48);

#[derive(Debug)]
pub struct CudaMamba2VerifyState {
    backend: CudaBackend,
    kernels: Mamba2Kernels,
    tokens: usize,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    next_history: DeviceBuffer<bf16>,
    /// The state after each of the first `tokens - 1` tokens, one apart.
    snapshots: DeviceBuffer<f32>,
    snapshot_histories: Vec<DeviceBuffer<bf16>>,
    convolved: DeviceBuffer<bf16>,
    scanned: DeviceBuffer<bf16>,
    source: Option<(u64, u64)>,
    destination: Option<RecurrentDestination>,
    identity: u64,
}

impl CudaBackend {
    /// A verification pass of `tokens` tokens, the fed token and its drafts.
    pub fn prepare_mamba2_verify_state(
        &self,
        geometry: Mamba2Geometry,
        tokens: usize,
    ) -> Result<CudaMamba2VerifyState> {
        if tokens < 2 {
            return Err(Error::InvalidDecoderKernel("Mamba-2 verification has no drafts"));
        }
        let (stream, pool) = (&self.inner.stream, &self.inner.pool);
        let (state, history) = (geometry.state_elements()?, geometry.history_elements()?);
        let rows = |elements: usize| {
            elements
                .checked_mul(tokens)
                .ok_or(Error::InvalidDecoderKernel("Mamba-2 verification size overflow"))
        };
        Ok(CudaMamba2VerifyState {
            backend: self.clone(),
            kernels: Mamba2Kernels::compile(&self.inner.compiler, geometry)?,
            tokens,
            state: pool.allocate(stream, state)?,
            history: pool.allocate(stream, history)?,
            next_history: pool.allocate(stream, history)?,
            snapshots: pool.allocate(stream, state * (tokens - 1))?,
            snapshot_histories: (1..tokens)
                .map(|_| pool.allocate(stream, history))
                .collect::<std::result::Result<_, _>>()?,
            convolved: pool.allocate(stream, rows(geometry.convolution_channels()?)?)?,
            scanned: pool.allocate(stream, rows(geometry.inner()?)?)?,
            source: None,
            destination: None,
            identity: NEXT_VERIFY_IDENTITY.fetch_add(1, Ordering::Relaxed),
        })
    }
}

impl CudaMamba2VerifyState {
    /// Places `state` here unless it is already resident and unchanged; a
    /// displaced sequence is first written back to its own buffers.
    pub fn pack(&mut self, state: &CudaMamba2State) -> Result<()> {
        if state.geometry() != self.kernels.geometry() {
            return Err(Error::InvalidDecoderKernel("Mamba-2 verification geometry differs"));
        }
        if self.source == Some(state.stamp()) && state.resident_in(self.identity, 0) {
            return Ok(());
        }
        let stream = &self.backend.inner.stream;
        if let Some(destination) = self.destination.take() {
            destination.preserve(stream, &self.state, &self.history)?;
        }
        let (source, range) = state.state_source();
        stream.copy_device_range(source, range, &mut self.state, 0)?;
        let (source, range) = state.history_source();
        stream.copy_device_range(source, range, &mut self.history, 0)?;
        self.source = None;
        Ok(())
    }

    /// Device work of the pass over `projection`; no allocation or host
    /// synchronization.
    pub fn execute(
        &mut self,
        projection: &DeviceBuffer<bf16>,
        parameters: Mamba2Parameters<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        let mut histories = self.snapshot_histories.iter_mut().collect::<Vec<_>>();
        let mut launch = Mamba2Launch {
            tokens: self.tokens,
            projection,
            parameters,
            state: &mut self.state,
            history: &self.history,
            next_history: &mut self.next_history,
            convolved: &mut self.convolved,
            scanned: &mut self.scanned,
            output,
        };
        self.kernels.execute_snapshotted(
            stream,
            &mut launch,
            &mut self.snapshots,
            &mut histories,
        )?;
        // A graph replays fixed addresses, so the history moves back in place.
        let length = self.history.len();
        stream.copy_device_range(&self.next_history, 0..length, &mut self.history, 0)?;
        Ok(())
    }

    /// Keeps the state after `kept` tokens of the last pass, which `state`
    /// started at its offset, and records it resident here.
    pub fn commit(&mut self, state: &mut CudaMamba2State, kept: usize) -> Result<()> {
        if !(1..=self.tokens).contains(&kept) {
            return Err(Error::InvalidDecoderKernel("Mamba-2 verification keeps no token"));
        }
        if kept < self.tokens {
            let stream = &self.backend.inner.stream;
            let length = self.state.len();
            let start = (kept - 1) * length;
            stream.copy_device_range(&self.snapshots, start..start + length, &mut self.state, 0)?;
            let history = &self.snapshot_histories[kept - 1];
            stream.copy_device_range(history, 0..history.len(), &mut self.history, 0)?;
        }
        state.advance(kept)?;
        self.destination =
            Some(state.bind_resident(self.identity, 0, (self.state.clone(), self.history.clone())));
        self.source = Some(state.stamp());
        Ok(())
    }
}
