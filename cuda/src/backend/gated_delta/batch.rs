use std::sync::atomic::{AtomicU64, Ordering};

use mircuda::{DeviceBuffer, bf16};

use super::{CudaGatedDeltaState, GatedDeltaInputs, GatedDeltaStateConfig, channels};
use crate::{
    CudaBackend, Error, Result,
    kernels::{
        GatedDeltaBatchConvolution, GatedDeltaBatchConvolutionSpec, GatedDeltaBatchRecurrence,
        GatedDeltaBatchSpec, GatedDeltaKernelInputs,
    },
};

#[derive(Debug)]
pub struct CudaGatedDeltaBatchState {
    backend: CudaBackend,
    config: GatedDeltaStateConfig,
    rows: usize,
    tokens: usize,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    convolution: GatedDeltaBatchConvolution,
    recurrence: GatedDeltaBatchRecurrence,
    sources: Vec<(u64, u64)>,
    identity: u64,
}

static NEXT_BATCH_IDENTITY: AtomicU64 = AtomicU64::new(1);

impl CudaGatedDeltaBatchState {
    pub(crate) fn new(
        backend: &CudaBackend,
        config: GatedDeltaStateConfig,
        rows: usize,
        tokens: usize,
    ) -> Result<Self> {
        let channels = channels(config)?;
        let state_per_row = state_elements(config)?;
        let history_per_row = history_elements(config)?;
        let allocate_bf16 = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            backend: backend.clone(),
            config,
            rows,
            tokens,
            state: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, checked(rows, state_per_row)?)?,
            history: allocate_bf16(checked(rows, history_per_row)?)?,
            convolution: GatedDeltaBatchConvolution::compile(
                &backend.inner.compiler,
                GatedDeltaBatchConvolutionSpec {
                    rows,
                    tokens,
                    channels,
                    kernel_size: config.convolution_kernel_size,
                },
            )?,
            recurrence: GatedDeltaBatchRecurrence::compile(
                &backend.inner.compiler,
                GatedDeltaBatchSpec {
                    rows,
                    tokens,
                    key_heads: config.key_heads,
                    value_heads: config.value_heads,
                    key_dim: config.key_dim,
                    value_dim: config.value_dim,
                },
            )?,
            sources: Vec::new(),
            identity: NEXT_BATCH_IDENTITY.fetch_add(1, Ordering::Relaxed),
        })
    }

    pub(crate) fn supports(&self, rows: usize, tokens: usize) -> bool {
        self.rows == rows && self.tokens == tokens
    }

    pub(crate) fn pack(&mut self, states: &[&mut CudaGatedDeltaState]) -> Result<()> {
        if states.len() != self.rows {
            return Err(Error::InvalidDecoderKernel("Gated Delta packed state row mismatch"));
        }
        if self.sources.len() == states.len()
            && self.sources.iter().zip(states).all(|(source, state)| *source == state.stamp())
        {
            return Ok(());
        }
        let stream = &self.backend.inner.stream;
        for (row, state) in states.iter().enumerate() {
            if state.config != self.config {
                return Err(Error::InvalidDecoderKernel(
                    "Gated Delta packed state config mismatch",
                ));
            }
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
        self.convolution.execute_in_place_strided(
            &self.backend.inner.stream,
            input,
            weight,
            &mut self.history,
            output,
            input_stride,
            input_offset,
        )
    }

    pub(crate) fn recur(
        &mut self,
        inputs: GatedDeltaInputs<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.recurrence.execute(
            &self.backend.inner.stream,
            GatedDeltaKernelInputs {
                query: inputs.query,
                key: inputs.key,
                value: inputs.value,
                alpha: inputs.alpha,
                beta: inputs.beta,
                a_log: inputs.a_log,
                dt_bias: inputs.dt_bias,
            },
            &mut self.state,
            output,
        )
    }

    pub(crate) fn commit(&mut self, states: &mut [&mut CudaGatedDeltaState]) -> Result<()> {
        for (row, state) in states.iter_mut().enumerate() {
            state.advance(self.tokens)?;
            state.bind_resident(self.identity, row, self.state.clone(), self.history.clone());
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
