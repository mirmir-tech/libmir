mod execution;
mod scratch;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod weights;

pub use execution::CudaAffineGatedDeltaExecution;
use models::layout::LinearAttentionConfig;
pub use weights::AffineGatedDeltaLayerWeights;

use super::{CudaGatedDeltaState, GatedDeltaStateConfig};
use crate::{CudaBackend, CudaTensorSet, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineGatedDeltaLayerConfig {
    pub hidden_size: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub convolution_kernel_size: usize,
    pub group_size: usize,
    pub bits: usize,
    pub rms_norm_epsilon: f32,
    pub norm_weight_shift: f32,
}

impl AffineGatedDeltaLayerConfig {
    pub fn from_linear_attention(
        hidden_size: usize,
        linear: &LinearAttentionConfig,
        group_size: usize,
        bits: usize,
        rms_norm_epsilon: f64,
        norm_weight_shift: f32,
    ) -> Result<Self> {
        let config = Self {
            hidden_size,
            key_heads: linear.key_heads,
            value_heads: linear.value_heads,
            key_dim: linear.key_head_dim,
            value_dim: linear.value_head_dim,
            convolution_kernel_size: linear.convolution_kernel_size,
            group_size,
            bits,
            rms_norm_epsilon: rms_norm_epsilon.to_string().parse()?,
            norm_weight_shift,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn state(self) -> Result<GatedDeltaStateConfig> {
        self.validate()?;
        Ok(GatedDeltaStateConfig {
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            convolution_kernel_size: self.convolution_kernel_size,
        })
    }

    pub(super) fn key_width(self) -> Result<usize> {
        checked(self.key_heads, self.key_dim)
    }

    pub(super) fn value_width(self) -> Result<usize> {
        checked(self.value_heads, self.value_dim)
    }

    pub(super) fn mixed_width(self) -> Result<usize> {
        let value_width = self.value_width()?;
        self.key_width()?
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or(Error::InvalidDecoderKernel("Gated Delta projection width overflow"))
    }

    fn validate(self) -> Result<()> {
        let state = GatedDeltaStateConfig {
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            convolution_kernel_size: self.convolution_kernel_size,
        };
        if self.hidden_size == 0
            || !valid_storage(self.group_size, self.bits)
            || (self.group_size > 0
                && (!self.hidden_size.is_multiple_of(self.group_size)
                    || !self.value_width()?.is_multiple_of(self.group_size)))
            || !self.rms_norm_epsilon.is_finite()
            || self.rms_norm_epsilon < 0.0
            || !self.norm_weight_shift.is_finite()
            || state.value_heads == 0
            || state.key_heads == 0
            || !state.value_heads.is_multiple_of(state.key_heads)
            || state.key_dim == 0
            || !state.key_dim.is_multiple_of(32)
            || state.value_dim == 0
            || state.convolution_kernel_size == 0
        {
            return Err(Error::InvalidDecoderKernel("invalid affine Gated Delta layer config"));
        }
        let _mixed = self.mixed_width()?;
        Ok(())
    }
}

const fn valid_storage(group_size: usize, bits: usize) -> bool {
    (group_size == 0 && bits == 0) || (group_size > 0 && matches!(bits, 2 | 3 | 4 | 5 | 6 | 8))
}

#[derive(Clone, Debug)]
pub struct CudaAffineGatedDeltaLayer {
    backend: CudaBackend,
    config: AffineGatedDeltaLayerConfig,
    weights: AffineGatedDeltaLayerWeights,
}

impl CudaAffineGatedDeltaLayer {
    pub fn from_tensors(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        prefix: &str,
        config: AffineGatedDeltaLayerConfig,
    ) -> Result<Self> {
        Self::new(backend, config, AffineGatedDeltaLayerWeights::load(tensors, prefix)?)
    }

    pub fn new(
        backend: &CudaBackend,
        config: AffineGatedDeltaLayerConfig,
        weights: AffineGatedDeltaLayerWeights,
    ) -> Result<Self> {
        config.validate()?;
        weights.validate(config)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            weights,
        })
    }

    pub fn prepare(&self, tokens: usize) -> Result<CudaAffineGatedDeltaExecution> {
        CudaAffineGatedDeltaExecution::new(&self.backend, self.config, &self.weights, tokens)
    }

    pub fn prepare_state(&self) -> Result<CudaGatedDeltaState> {
        self.backend.prepare_gated_delta_state(self.config.state()?)
    }
}

pub(super) fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("Gated Delta shape overflow"))
}
