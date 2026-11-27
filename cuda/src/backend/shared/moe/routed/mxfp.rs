use super::SharedRoutedExecution;
use crate::{
    CudaBackend, Result,
    backend::{
        linear::{MxFp4ExpertWeights, MxFp4GatheredMoeBf16},
        router::{BoundRouter, RouterWeights},
        shared::moe::{AffineSharedExpertMoeConfig, weights::AffineSharedExpertMoeWeights},
    },
    kernels::RouterUnitSpec,
};

pub(super) fn mxfp4(
    backend: &CudaBackend,
    config: AffineSharedExpertMoeConfig,
    weights: &AffineSharedExpertMoeWeights,
    experts: &MxFp4ExpertWeights,
    tokens: usize,
) -> Result<SharedRoutedExecution> {
    Ok(SharedRoutedExecution::MxFp4 {
        router: router(backend, config, weights, tokens)?,
        experts: Box::new(MxFp4GatheredMoeBf16::new(
            backend,
            tokens,
            config.top_k,
            config.activation,
            experts,
        )?),
    })
}

pub(super) fn mxfp8(
    backend: &CudaBackend,
    config: AffineSharedExpertMoeConfig,
    weights: &AffineSharedExpertMoeWeights,
    experts: &crate::backend::linear::MxFp8ExpertWeights,
    tokens: usize,
) -> Result<SharedRoutedExecution> {
    Ok(SharedRoutedExecution::MxFp8 {
        router: router(backend, config, weights, tokens)?,
        experts: Box::new(crate::backend::linear::MxFp8GatheredMoeBf16::new(
            backend,
            tokens,
            config.top_k,
            config.activation,
            experts,
        )?),
    })
}

/// The layer's router, bound to its projection and correction bias.
pub(super) fn router(
    backend: &CudaBackend,
    config: AffineSharedExpertMoeConfig,
    weights: &AffineSharedExpertMoeWeights,
    tokens: usize,
) -> Result<BoundRouter> {
    BoundRouter::new(
        backend,
        (
            RouterUnitSpec {
                tokens,
                experts: config.expert_count,
                top_k: config.top_k,
            },
            config.hidden_size,
        ),
        config.routing,
        RouterWeights {
            projection: &weights.router,
            correction_bias: weights.router_correction_bias.as_ref(),
        },
    )
}
