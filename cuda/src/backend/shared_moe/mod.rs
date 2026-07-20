mod execution;
mod scratch;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod weights;

pub use execution::CudaAffineSharedExpertMoeExecution;
pub use weights::AffineSharedExpertMoeWeights;

use crate::{CudaBackend, CudaTensorSet, Error, GatedActivation, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffineSharedExpertMoeConfig {
    pub hidden_size: usize,
    pub routed_intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub expert_count: usize,
    pub top_k: usize,
    pub group_size: usize,
    pub expert_bits: usize,
    pub router_bits: usize,
    pub activation: GatedActivation,
}

impl AffineSharedExpertMoeConfig {
    fn validate(self) -> Result<()> {
        if self.hidden_size == 0
            || self.routed_intermediate_size == 0
            || self.shared_intermediate_size == 0
            || self.expert_count == 0
            || self.expert_count > 256
            || self.top_k == 0
            || self.top_k > self.expert_count
            || self.group_size == 0
            || !self.hidden_size.is_multiple_of(self.group_size)
            || !self.routed_intermediate_size.is_multiple_of(self.group_size)
            || !self.shared_intermediate_size.is_multiple_of(self.group_size)
            || !matches!(self.expert_bits, 4 | 8)
            || !matches!(self.router_bits, 4 | 8)
        {
            return Err(Error::InvalidDecoderKernel("invalid affine shared-expert MoE config"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CudaAffineSharedExpertMoe {
    backend: CudaBackend,
    config: AffineSharedExpertMoeConfig,
    weights: AffineSharedExpertMoeWeights,
}

impl CudaAffineSharedExpertMoe {
    pub fn from_tensors(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        prefix: &str,
        config: AffineSharedExpertMoeConfig,
    ) -> Result<Self> {
        Self::new(backend, config, AffineSharedExpertMoeWeights::load(tensors, prefix)?)
    }

    pub fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        weights: AffineSharedExpertMoeWeights,
    ) -> Result<Self> {
        config.validate()?;
        weights.validate(config)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            weights,
        })
    }

    pub fn prepare(&self, tokens: usize) -> Result<CudaAffineSharedExpertMoeExecution> {
        CudaAffineSharedExpertMoeExecution::new(&self.backend, self.config, &self.weights, tokens)
    }
}

pub(super) fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("affine shared-expert MoE shape overflow"))
}
