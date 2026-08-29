mod execution;
mod gate_up;
mod routed;
mod scratch;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod weights;

pub use execution::CudaAffineSharedExpertMoeExecution;
pub use weights::AffineSharedExpertMoeWeights;

use crate::{CudaBackend, CudaTensorSet, Error, ExecutionPhase, GatedActivation, Result};

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
            || !valid_storage(self.group_size, self.expert_bits, self.router_bits)
            || (self.group_size > 0
                && (!self.hidden_size.is_multiple_of(self.group_size)
                    || !self.routed_intermediate_size.is_multiple_of(self.group_size)
                    || !self.shared_intermediate_size.is_multiple_of(self.group_size)))
        {
            return Err(Error::InvalidDecoderKernel("invalid affine shared-expert MoE config"));
        }
        Ok(())
    }
}

const fn valid_storage(group_size: usize, expert_bits: usize, router_bits: usize) -> bool {
    (group_size == 0 && expert_bits == 0 && router_bits == 0)
        || (group_size > 0
            && matches!(expert_bits, 2 | 3 | 4 | 5 | 6 | 8)
            && matches!(router_bits, 2 | 3 | 4 | 5 | 6 | 8))
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
        let phase = if tokens == 1 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        self.prepare_phase(tokens, phase)
    }

    pub(crate) fn prepare_phase(
        &self,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<CudaAffineSharedExpertMoeExecution> {
        CudaAffineSharedExpertMoeExecution::new(
            &self.backend, self.config, &self.weights, tokens, phase,
        )
    }
}

pub(super) fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("affine shared-expert MoE shape overflow"))
}
