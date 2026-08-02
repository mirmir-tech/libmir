use super::SharedRoutedExecution;
use crate::{
    CudaBackend, DenseRole, Result,
    backend::{
        linear::{CheckpointProjection, MxFp4ExpertWeights, MxFp4GatheredMoeBf16},
        shared_moe::{AffineSharedExpertMoeConfig, weights::AffineSharedExpertMoeWeights},
    },
    kernels::{RouterUnitSpec, RouterUnitTopK},
};

pub(super) fn mxfp4(
    backend: &CudaBackend,
    config: AffineSharedExpertMoeConfig,
    weights: &AffineSharedExpertMoeWeights,
    experts: &MxFp4ExpertWeights,
    tokens: usize,
) -> Result<SharedRoutedExecution> {
    let (router, top_k, scores, selected, routing) = routing(backend, config, weights, tokens, 4)?;
    Ok(SharedRoutedExecution::MxFp4 {
        router,
        top_k,
        scores,
        selected,
        routing,
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
    let (router, top_k, scores, selected, routing) = routing(backend, config, weights, tokens, 8)?;
    Ok(SharedRoutedExecution::MxFp8 {
        router,
        top_k,
        scores,
        selected,
        routing,
        experts: Box::new(crate::backend::linear::MxFp8GatheredMoeBf16::new(
            backend,
            tokens,
            config.top_k,
            config.activation,
            experts,
        )?),
    })
}

type Routing = (
    Box<CheckpointProjection>,
    RouterUnitTopK,
    mircuda::DeviceBuffer<mircuda::bf16>,
    mircuda::DeviceBuffer<u32>,
    mircuda::DeviceBuffer<mircuda::bf16>,
);

pub(super) fn routing(
    backend: &CudaBackend,
    config: AffineSharedExpertMoeConfig,
    weights: &AffineSharedExpertMoeWeights,
    tokens: usize,
    bits: usize,
) -> Result<Routing> {
    let selections = tokens.checked_mul(config.top_k).ok_or(crate::Error::InvalidDecoderKernel(
        if bits == 4 {
            "MXFP4 shared routing size overflow"
        } else {
            "MXFP8 shared routing size overflow"
        },
    ))?;
    Ok((
        Box::new(CheckpointProjection::new(
            backend,
            tokens,
            config.hidden_size,
            config.expert_count,
            DenseRole::Router,
            &weights.router,
        )?),
        RouterUnitTopK::compile(
            backend.compiler(),
            RouterUnitSpec {
                tokens,
                experts: config.expert_count,
                top_k: config.top_k,
            },
        )?,
        backend.pool().allocate(backend.stream(), tokens * config.expert_count)?,
        backend.pool().allocate(backend.stream(), selections)?,
        backend.pool().allocate(backend.stream(), selections)?,
    ))
}
