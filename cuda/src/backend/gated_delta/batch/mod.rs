use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use mircuda::{DeviceBuffer, bf16};

use super::{
    CudaGatedDeltaState, GatedDeltaInputs, GatedDeltaStateConfig, channels,
    residency::GatedDeltaDestination,
};
use crate::{CudaBackend, Error, Result, kernels::GatedDeltaKernelInputs};

mod kernels;
use kernels::BatchKernels;

/// Identifies the packed states one layer shares between its batches of
/// every row count: the layer's weights and the tokens per row.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct GatedDeltaBatchKey {
    pub(in crate::backend) layer: u64,
    pub(in crate::backend) config: GatedDeltaStateConfig,
    pub(in crate::backend) tokens: usize,
}

/// Packed recurrent states of up to `capacity` rows. One decode batch
/// executes at a time, so the batches of every row count share these
/// buffers per layer; a session's state may stay resident here between
/// steps (`residency`) and is preserved before another row overwrites it.
#[derive(Debug)]
pub struct CudaGatedDeltaBatchState {
    backend: CudaBackend,
    config: GatedDeltaStateConfig,
    capacity: usize,
    rows: usize,
    tokens: usize,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    kernels: HashMap<usize, BatchKernels>,
    sources: Vec<(u64, u64)>,
    destinations: Vec<GatedDeltaDestination>,
    identity: u64,
}

static NEXT_BATCH_IDENTITY: AtomicU64 = AtomicU64::new(1);

impl CudaGatedDeltaBatchState {
    pub(crate) fn new(
        backend: &CudaBackend,
        config: GatedDeltaStateConfig,
        capacity: usize,
        tokens: usize,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::InvalidDecoderKernel("Gated Delta packed state has no rows"));
        }
        let state_per_row = state_elements(config)?;
        let history_per_row = history_elements(config)?;
        let allocate_bf16 = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            backend: backend.clone(),
            config,
            capacity,
            rows: 0,
            tokens,
            state: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, checked(capacity, state_per_row)?)?,
            history: allocate_bf16(checked(capacity, history_per_row)?)?,
            kernels: HashMap::new(),
            sources: Vec::new(),
            destinations: Vec::new(),
            identity: NEXT_BATCH_IDENTITY.fetch_add(1, Ordering::Relaxed),
        })
    }

    pub(crate) fn supports(&self, rows: usize, tokens: usize) -> bool {
        rows <= self.capacity && self.tokens == tokens
    }

    pub(crate) fn pack(&mut self, states: &[&mut CudaGatedDeltaState]) -> Result<()> {
        if states.is_empty()
            || states.len() > self.capacity
            || states.iter().any(|state| state.config != self.config)
        {
            return Err(Error::InvalidDecoderKernel("Gated Delta packed state row mismatch"));
        }
        self.rows = states.len();
        self.kernels(states.len())?;
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
        // Preserve every live previous row before writing any replacement. A
        // permutation aliases this buffer, and omitted sessions may resume
        // later.
        for destination in &self.destinations {
            destination.preserve(stream, &self.state, &self.history)?;
        }
        self.destinations.clear();
        for (row, state) in states.iter().enumerate() {
            if !state.resident_in(self.identity, row) {
                let (source, range) = state.state_source();
                stream.copy_device_range(
                    source,
                    range,
                    &mut self.state,
                    checked(row, state.state.len())?,
                )?;
                let (source, range) = state.history_source();
                stream.copy_device_range(
                    source,
                    range,
                    &mut self.history,
                    checked(row, state.convolution.len())?,
                )?;
            }
        }
        self.sources = states.iter().map(|state| state.stamp()).collect();
        Ok(())
    }

    pub(crate) fn convolve(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.convolve_strided(input, weight, output, channels(self.config)?, 0)
    }

    pub(crate) fn convolve_strided(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        input_stride: usize,
        input_offset: usize,
    ) -> Result<()> {
        let stream = self.backend.inner.stream.clone();
        let rows = self.rows;
        let Self { history, kernels, .. } = self;
        kernels
            .get_mut(&rows)
            .ok_or(Error::InvalidDecoderKernel("Gated Delta packed state was not packed"))?
            .convolution
            .execute_in_place_strided(
                &stream, input, weight, history, output, input_stride, input_offset,
            )
    }

    pub(crate) fn recur(
        &mut self,
        inputs: GatedDeltaInputs<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let stream = self.backend.inner.stream.clone();
        let rows = self.rows;
        let Self { state, kernels, .. } = self;
        let recurrence = &kernels
            .get(&rows)
            .ok_or(Error::InvalidDecoderKernel("Gated Delta packed state was not packed"))?
            .recurrence;
        recurrence.execute(
            &stream,
            GatedDeltaKernelInputs {
                query: inputs.query,
                key: inputs.key,
                value: inputs.value,
                alpha: inputs.alpha,
                beta: inputs.beta,
                a_log: inputs.a_log,
                dt_bias: inputs.dt_bias,
            },
            state,
            output,
        )
    }

    pub(crate) fn commit(&mut self, states: &mut [&mut CudaGatedDeltaState]) -> Result<()> {
        self.destinations.clear();
        for (row, state) in states.iter_mut().enumerate() {
            state.advance(self.tokens)?;
            self.destinations.push(state.bind_resident(
                self.identity,
                row,
                self.state.clone(),
                self.history.clone(),
            ));
            self.sources[row] = state.stamp();
        }
        Ok(())
    }
}

fn state_elements(config: GatedDeltaStateConfig) -> Result<usize> {
    checked(checked(config.value_heads, config.value_dim)?, config.key_dim)
}

fn history_elements(config: GatedDeltaStateConfig) -> Result<usize> {
    checked(config.convolution_kernel_size - 1, channels(config)?)
}

fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("Gated Delta packed state size overflow"))
}
