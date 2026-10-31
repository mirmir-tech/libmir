//! Latent routed-expert block: sigmoid routing with correction-bias
//! selection, routed experts in a projected latent space with an ungated
//! `relu²` activation, and an optional full-width shared expert.
//!
//! `out = fc2(Σ_e w_e · down_e(relu²(up_e(fc1(x))))) + shared(x)`

mod execution;
mod scratch;
mod weights;

pub use execution::CudaLatentMoeExecution;
use mircuda::MarlinNvFp4ThreadConfig;
pub(in crate::backend) use scratch::{LatentMoeScratch, LatentMoeScratchKey};
pub use weights::LatentMoeWeights;

use crate::{CudaBackend, Error, Result};

/// Geometry and routing scale of one latent routed-expert block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatentMoeConfig {
    pub hidden: usize,
    pub latent: usize,
    pub experts: usize,
    pub top_k: usize,
    pub expert_intermediate: usize,
    /// Intermediate width of the shared expert, if the block has one.
    pub shared_intermediate: Option<usize>,
    pub routed_scale: f32,
}

impl LatentMoeConfig {
    /// Block geometry of a state-space hybrid decoder's routed blocks.
    #[must_use]
    pub fn from_hybrid(hidden: usize, routed: &models::layout::LatentRoutedConfig) -> Self {
        Self {
            hidden,
            latent: routed.latent_size,
            experts: routed.expert_count,
            top_k: routed.top_k,
            expert_intermediate: routed.expert_intermediate_size,
            shared_intermediate: routed.shared_intermediate_size,
            // Kernels apply the scale in FP32; checkpoint values are small
            // positive constants that FP32 represents to full precision.
            #[allow(clippy::cast_possible_truncation)]
            routed_scale: routed.routed_scale as f32,
        }
    }

    fn validate(self) -> Result<()> {
        let valid = [self.hidden, self.latent, self.experts, self.top_k, self.expert_intermediate]
            .into_iter()
            .all(|value| value > 0)
            && self.shared_intermediate != Some(0)
            && self.top_k <= self.experts
            && self.routed_scale.is_finite()
            && self.routed_scale > 0.0;
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidDecoderKernel("invalid latent routed-expert configuration"))
        }
    }
}

/// Device-resident latent routed-expert block.
#[derive(Clone, Debug)]
pub struct CudaLatentMoe {
    backend: CudaBackend,
    config: LatentMoeConfig,
    weights: LatentMoeWeights,
}

impl CudaLatentMoe {
    pub fn new(
        backend: &CudaBackend,
        config: LatentMoeConfig,
        weights: LatentMoeWeights,
    ) -> Result<Self> {
        config.validate()?;
        weights.validate(config)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            weights,
        })
    }

    #[must_use]
    pub const fn config(&self) -> LatentMoeConfig {
        self.config
    }

    /// Prepares kernels and scratch for chunks of `tokens` rows.
    pub fn prepare(&self, tokens: usize) -> Result<CudaLatentMoeExecution> {
        self.prepare_with(tokens, expert_tiles(tokens, self.config))
    }

    /// Like [`Self::prepare`], with an explicit Marlin expert tile.
    pub(crate) fn prepare_with(
        &self,
        tokens: usize,
        tiles: MarlinNvFp4ThreadConfig,
    ) -> Result<CudaLatentMoeExecution> {
        CudaLatentMoeExecution::new(&self.backend, self.config, &self.weights, tokens, tiles)
    }
}

/// Marlin tile for the routed experts, chosen by the average number of rows
/// each expert receives; thresholds measured on GB10 with Nemotron-3.5 Super
/// layer weights (see the NH-300 report).
const fn expert_tiles(tokens: usize, config: LatentMoeConfig) -> MarlinNvFp4ThreadConfig {
    let rows_per_expert = tokens.saturating_mul(config.top_k) / config.experts;
    if rows_per_expert < 8 {
        MarlinNvFp4ThreadConfig::N128K128
    } else if rows_per_expert < 64 {
        MarlinNvFp4ThreadConfig::N128K64
    } else {
        MarlinNvFp4ThreadConfig::M64N128K64
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests;
