use mircuda::{DeviceBuffer, bf16};

use super::CudaBackend;
use crate::{
    Result,
    kernels::{
        GatedDeltaConvolution, GatedDeltaConvolutionSpec, GatedDeltaLaunch, GatedDeltaRecurrence,
        GatedDeltaSpec,
    },
};

mod layer;
pub use layer::{
    AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights, CudaAffineGatedDeltaExecution,
    CudaAffineGatedDeltaLayer,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedDeltaStateConfig {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub convolution_kernel_size: usize,
}

#[derive(Clone, Copy)]
pub struct GatedDeltaInputs<'a> {
    pub query: &'a DeviceBuffer<bf16>,
    pub key: &'a DeviceBuffer<bf16>,
    pub value: &'a DeviceBuffer<bf16>,
    pub alpha: &'a DeviceBuffer<bf16>,
    pub beta: &'a DeviceBuffer<bf16>,
    pub a_log: &'a DeviceBuffer<bf16>,
    pub dt_bias: &'a DeviceBuffer<bf16>,
}

#[derive(Debug)]
pub struct CudaGatedDeltaState {
    backend: CudaBackend,
    config: GatedDeltaStateConfig,
    state: DeviceBuffer<f32>,
    convolution: DeviceBuffer<bf16>,
    next_convolution: DeviceBuffer<bf16>,
    offset: usize,
}

impl CudaBackend {
    pub fn prepare_gated_delta_state(
        &self,
        config: GatedDeltaStateConfig,
    ) -> Result<CudaGatedDeltaState> {
        let probe = GatedDeltaRecurrence::compile(
            &self.inner.compiler,
            GatedDeltaSpec {
                tokens: 1,
                key_heads: config.key_heads,
                value_heads: config.value_heads,
                key_dim: config.key_dim,
                value_dim: config.value_dim,
            },
        )?;
        let channels = channels(config)?;
        let convolution = GatedDeltaConvolution::compile(
            &self.inner.compiler,
            GatedDeltaConvolutionSpec {
                tokens: 1,
                channels,
                kernel_size: config.convolution_kernel_size,
            },
        )?;
        let history = convolution.history_elements()?;
        Ok(CudaGatedDeltaState {
            backend: self.clone(),
            config,
            state: self.inner.pool.allocate_zeroed(&self.inner.stream, probe.state_elements()?)?,
            convolution: self.inner.pool.allocate_zeroed(&self.inner.stream, history)?,
            next_convolution: self.inner.pool.allocate_zeroed(&self.inner.stream, history)?,
            offset: 0,
        })
    }
}

impl CudaGatedDeltaState {
    #[must_use]
    pub const fn config(&self) -> GatedDeltaStateConfig {
        self.config
    }

    pub fn convolve_silu(
        &mut self,
        tokens: usize,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let operation = GatedDeltaConvolution::compile(
            &self.backend.inner.compiler,
            GatedDeltaConvolutionSpec {
                tokens,
                channels: channels(self.config)?,
                kernel_size: self.config.convolution_kernel_size,
            },
        )?;
        operation.execute(
            &self.backend.inner.stream,
            input,
            weight,
            &self.convolution,
            &mut self.next_convolution,
            output,
        )?;
        std::mem::swap(&mut self.convolution, &mut self.next_convolution);
        Ok(())
    }

    pub fn execute(
        &mut self,
        tokens: usize,
        inputs: GatedDeltaInputs<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let operation = GatedDeltaRecurrence::compile(
            &self.backend.inner.compiler,
            GatedDeltaSpec {
                tokens,
                key_heads: self.config.key_heads,
                value_heads: self.config.value_heads,
                key_dim: self.config.key_dim,
                value_dim: self.config.value_dim,
            },
        )?;
        operation.execute(
            &self.backend.inner.stream,
            &mut GatedDeltaLaunch {
                query: inputs.query,
                key: inputs.key,
                value: inputs.value,
                alpha: inputs.alpha,
                beta: inputs.beta,
                a_log: inputs.a_log,
                dt_bias: inputs.dt_bias,
                state: &mut self.state,
                output,
            },
        )?;
        self.offset = self
            .offset
            .checked_add(tokens)
            .ok_or(crate::Error::InvalidDecoderKernel("Gated Delta state offset overflow"))?;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.state = self
            .backend
            .inner
            .pool
            .allocate_zeroed(&self.backend.inner.stream, self.state.len())?;
        let history = self.convolution.len();
        self.convolution =
            self.backend.inner.pool.allocate_zeroed(&self.backend.inner.stream, history)?;
        self.next_convolution =
            self.backend.inner.pool.allocate_zeroed(&self.backend.inner.stream, history)?;
        self.offset = 0;
        Ok(())
    }

    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }
}

fn channels(config: GatedDeltaStateConfig) -> Result<usize> {
    let key = config
        .key_heads
        .checked_mul(config.key_dim)
        .and_then(|width| width.checked_mul(2))
        .ok_or(crate::Error::InvalidDecoderKernel("Gated Delta key width overflow"))?;
    let value = config
        .value_heads
        .checked_mul(config.value_dim)
        .ok_or(crate::Error::InvalidDecoderKernel("Gated Delta value width overflow"))?;
    key.checked_add(value)
        .ok_or(crate::Error::InvalidDecoderKernel("Gated Delta channel width overflow"))
}

#[cfg(all(test, target_os = "linux"))]
mod tests;
