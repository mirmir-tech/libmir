use std::sync::atomic::{AtomicU64, Ordering};

use mircuda::{DeviceBuffer, bf16};

use super::CudaBackend;
use crate::{
    Result,
    kernels::{
        GatedDeltaConvolution, GatedDeltaConvolutionSpec, GatedDeltaLaunch, GatedDeltaRecurrence,
        GatedDeltaSpec,
    },
};

mod batch;
mod checkpoint;
mod convolution;
mod layer;
mod residency;
pub(super) mod workspace;
pub use batch::CudaGatedDeltaBatchState;
pub use checkpoint::CudaGatedDeltaCheckpoint;
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
    decay: DeviceBuffer<f32>,
    update: DeviceBuffer<f32>,
    offset: usize,
    identity: u64,
    revision: u64,
    resident: Option<residency::GatedDeltaResidency>,
}

static NEXT_STATE_IDENTITY: AtomicU64 = AtomicU64::new(1);

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
            decay: self.inner.pool.allocate(&self.inner.stream, config.value_heads)?,
            update: self.inner.pool.allocate(&self.inner.stream, config.value_heads)?,
            offset: 0,
            identity: NEXT_STATE_IDENTITY.fetch_add(1, Ordering::Relaxed),
            revision: 0,
            resident: None,
        })
    }
}

impl CudaGatedDeltaState {
    #[must_use]
    pub const fn config(&self) -> GatedDeltaStateConfig {
        self.config
    }

    pub fn execute(
        &mut self,
        tokens: usize,
        inputs: GatedDeltaInputs<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.materialize()?;
        let gates = tokens
            .checked_mul(self.config.value_heads)
            .ok_or(crate::Error::InvalidDecoderKernel("Gated Delta gate size overflow"))?;
        if self.decay.len() != gates {
            self.decay = self.backend.inner.pool.allocate(&self.backend.inner.stream, gates)?;
            self.update = self.backend.inner.pool.allocate(&self.backend.inner.stream, gates)?;
        }
        let spec = GatedDeltaSpec {
            tokens,
            key_heads: self.config.key_heads,
            value_heads: self.config.value_heads,
            key_dim: self.config.key_dim,
            value_dim: self.config.value_dim,
        };
        let backend = self.backend.clone();
        let mut launch = GatedDeltaLaunch {
            query: inputs.query,
            key: inputs.key,
            value: inputs.value,
            alpha: inputs.alpha,
            beta: inputs.beta,
            a_log: inputs.a_log,
            dt_bias: inputs.dt_bias,
            decay: &mut self.decay,
            update: &mut self.update,
            state: &mut self.state,
            output,
        };
        if crate::kernels::GatedDeltaChunked::supports(
            backend.inner.device.compute_capability,
            spec,
        ) {
            backend.execute_gated_delta_chunked(spec, &mut launch)?;
        } else {
            GatedDeltaRecurrence::compile(&backend.inner.compiler, spec)?
                .execute(&backend.inner.stream, &mut launch)?;
        }
        self.offset = self
            .offset
            .checked_add(tokens)
            .ok_or(crate::Error::InvalidDecoderKernel("Gated Delta state offset overflow"))?;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.resident = None;
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
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) const fn stamp(&self) -> (u64, u64) {
        (self.identity, self.revision)
    }

    pub(crate) fn advance(&mut self, tokens: usize) -> Result<()> {
        self.offset = self
            .offset
            .checked_add(tokens)
            .ok_or(crate::Error::InvalidDecoderKernel("Gated Delta state offset overflow"))?;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
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
