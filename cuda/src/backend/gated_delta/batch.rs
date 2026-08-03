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
pub(crate) struct CudaGatedDeltaBatchState {
    backend: CudaBackend,
    config: GatedDeltaStateConfig,
    rows: usize,
    tokens: usize,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    next_history: DeviceBuffer<bf16>,
    convolution: GatedDeltaBatchConvolution,
    recurrence: GatedDeltaBatchRecurrence,
}

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
            next_history: allocate_bf16(checked(rows, history_per_row)?)?,
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
        })
    }

    pub(crate) fn supports(&self, rows: usize, tokens: usize) -> bool {
        self.rows == rows && self.tokens == tokens
    }

    pub(crate) fn pack(&mut self, states: &[&mut CudaGatedDeltaState]) -> Result<()> {
        if states.len() != self.rows {
            return Err(Error::InvalidDecoderKernel("Gated Delta packed state row mismatch"));
        }
        let stream = &self.backend.inner.stream;
        for (row, state) in states.iter().enumerate() {
            if state.config != self.config {
                return Err(Error::InvalidDecoderKernel(
                    "Gated Delta packed state config mismatch",
                ));
            }
            stream.copy_device_range(
                &state.state,
                0..state.state.len(),
                &mut self.state,
                checked(row, state.state.len())?,
            )?;
            stream.copy_device_range(
                &state.convolution,
                0..state.convolution.len(),
                &mut self.history,
                checked(row, state.convolution.len())?,
            )?;
        }
        Ok(())
    }

    pub(crate) fn convolve(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.convolution.execute(
            &self.backend.inner.stream,
            input,
            weight,
            &self.history,
            &mut self.next_history,
            output,
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

    pub(crate) fn commit(&self, states: &mut [&mut CudaGatedDeltaState]) -> Result<()> {
        let stream = &self.backend.inner.stream;
        for (row, state) in states.iter_mut().enumerate() {
            let state_len = state.state.len();
            let history_len = state.convolution.len();
            stream.copy_device_range(
                &self.state,
                checked(row, state_len)?..checked(row + 1, state_len)?,
                &mut state.state,
                0,
            )?;
            stream.copy_device_range(
                &self.next_history,
                checked(row, history_len)?..checked(row + 1, history_len)?,
                &mut state.convolution,
                0,
            )?;
            state.offset = state
                .offset
                .checked_add(self.tokens)
                .ok_or(Error::InvalidDecoderKernel("Gated Delta state offset overflow"))?;
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
