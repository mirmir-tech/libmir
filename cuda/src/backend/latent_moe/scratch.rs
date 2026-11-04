//! Rows a latent routed-expert block uses only while it executes. Blocks run
//! one after another, so every block of one shape shares a single set.

use mircuda::{DeviceBuffer, bf16};

use super::LatentMoeConfig;
use crate::{CudaBackend, Error, Result, backend::scratch_pool::SharedScratch};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct LatentMoeScratchKey {
    tokens: usize,
    hidden: usize,
    latent: usize,
    experts: usize,
    top_k: usize,
    shared_intermediate: Option<usize>,
}

#[derive(Debug)]
pub(in crate::backend) struct LatentMoeScratch {
    pub(super) logits: DeviceBuffer<f32>,
    pub(super) selected: DeviceBuffer<u32>,
    pub(super) routing: DeviceBuffer<bf16>,
    pub(super) latent: DeviceBuffer<bf16>,
    pub(super) routed_latent: DeviceBuffer<bf16>,
    pub(super) routed: DeviceBuffer<bf16>,
    pub(super) shared: Option<SharedExpertRows>,
}

#[derive(Debug)]
pub(in crate::backend) struct SharedExpertRows {
    pub(super) intermediate: DeviceBuffer<bf16>,
    pub(super) activated: DeviceBuffer<bf16>,
    pub(super) output: DeviceBuffer<bf16>,
}

impl CudaBackend {
    pub(super) fn latent_moe_scratch(
        &self,
        config: LatentMoeConfig,
        tokens: usize,
    ) -> Result<SharedScratch<LatentMoeScratch>> {
        let key = LatentMoeScratchKey {
            tokens,
            hidden: config.hidden,
            latent: config.latent,
            experts: config.experts,
            top_k: config.top_k,
            shared_intermediate: config.shared_intermediate,
        };
        let elements = |columns: usize| {
            tokens
                .checked_mul(columns)
                .ok_or(Error::InvalidDecoderKernel("latent MoE size overflow"))
        };
        let (stream, pool) = (&self.inner.stream, &self.inner.pool);
        let allocate = |columns| -> Result<DeviceBuffer<bf16>> {
            Ok(pool.allocate(stream, elements(columns)?)?)
        };
        self.inner.latent_moe_scratch.acquire(key, || {
            Ok(LatentMoeScratch {
                logits: pool.allocate(stream, elements(config.experts)?)?,
                selected: pool.allocate(stream, elements(config.top_k)?)?,
                routing: allocate(config.top_k)?,
                latent: allocate(config.latent)?,
                routed_latent: allocate(config.latent)?,
                routed: allocate(config.hidden)?,
                shared: config
                    .shared_intermediate
                    .map(|width| {
                        Ok::<_, Error>(SharedExpertRows {
                            intermediate: allocate(width)?,
                            activated: allocate(width)?,
                            output: allocate(config.hidden)?,
                        })
                    })
                    .transpose()?,
            })
        })
    }
}
