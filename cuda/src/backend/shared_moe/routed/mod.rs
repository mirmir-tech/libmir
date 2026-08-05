use mircuda::{DeviceBuffer, bf16};

use self::affine::AutoAffineRoutedExperts;
use super::{
    AffineSharedExpertMoeConfig,
    scratch::AffineSharedMoeScratch,
    weights::{AffineSharedExpertMoeWeights, RoutedSharedMoeWeights},
};
use crate::{
    AffineQuantizedConfig, AffineRouterBf16, CudaBackend, DenseRole, ExecutionPhase, Result,
    backend::linear::{CheckpointProjection, SelectedDenseMoeBf16},
    kernels::RouterUnitTopK,
};

mod affine;
mod mxfp;
mod nvfp4;
mod parallel;

#[derive(Debug)]
pub(super) enum SharedRoutedExecution {
    Affine {
        router: Box<AffineRouterBf16>,
        experts: Box<AutoAffineRoutedExperts>,
    },
    Dense {
        router: Box<CheckpointProjection>,
        top_k: RouterUnitTopK,
        scores: DeviceBuffer<bf16>,
        selected: DeviceBuffer<u32>,
        routing: DeviceBuffer<bf16>,
        experts: Box<SelectedDenseMoeBf16>,
        intermediate: DeviceBuffer<bf16>,
    },
    MxFp4 {
        router: Box<CheckpointProjection>,
        top_k: RouterUnitTopK,
        scores: DeviceBuffer<bf16>,
        selected: DeviceBuffer<u32>,
        routing: DeviceBuffer<bf16>,
        experts: Box<crate::backend::linear::MxFp4GatheredMoeBf16>,
    },
    MxFp8 {
        router: Box<CheckpointProjection>,
        top_k: RouterUnitTopK,
        scores: DeviceBuffer<bf16>,
        selected: DeviceBuffer<u32>,
        routing: DeviceBuffer<bf16>,
        experts: Box<crate::backend::linear::MxFp8GatheredMoeBf16>,
    },
    NvFp4(Box<nvfp4::NvFp4RoutedExecution>),
}

impl SharedRoutedExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        weights: &AffineSharedExpertMoeWeights,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        match &weights.routed {
            RoutedSharedMoeWeights::Affine(weights) => Ok(Self::Affine {
                router: Box::new(backend.prepare_affine_router_bf16(
                    tokens,
                    AffineQuantizedConfig::new(
                        config.hidden_size,
                        config.expert_count,
                        config.group_size,
                        config.router_bits,
                    ),
                    config.top_k,
                )?),
                experts: Box::new(AutoAffineRoutedExperts::new(backend, config, weights, tokens)?),
            }),
            RoutedSharedMoeWeights::Dense(experts) => {
                let selections = tokens.checked_mul(config.top_k).ok_or(
                    crate::Error::InvalidDecoderKernel("dense shared routing size overflow"),
                )?;
                Ok(Self::Dense {
                    router: Box::new(CheckpointProjection::new(
                        backend,
                        tokens,
                        config.hidden_size,
                        config.expert_count,
                        DenseRole::Router,
                        &weights.router,
                    )?),
                    top_k: RouterUnitTopK::compile(
                        backend.compiler(),
                        crate::kernels::RouterUnitSpec {
                            tokens,
                            experts: config.expert_count,
                            top_k: config.top_k,
                        },
                    )?,
                    scores: backend
                        .pool()
                        .allocate(backend.stream(), tokens * config.expert_count)?,
                    selected: backend.pool().allocate(backend.stream(), selections)?,
                    routing: backend.pool().allocate(backend.stream(), selections)?,
                    experts: Box::new(SelectedDenseMoeBf16::new(
                        backend,
                        tokens,
                        config.top_k,
                        experts,
                        config.activation.into(),
                    )?),
                    intermediate: backend.pool().allocate(
                        backend.stream(),
                        experts.intermediate_elements(tokens, config.top_k)?,
                    )?,
                })
            },
            RoutedSharedMoeWeights::MxFp4(experts) => {
                mxfp::mxfp4(backend, config, weights, experts, tokens)
            },
            RoutedSharedMoeWeights::MxFp8(experts) => {
                mxfp::mxfp8(backend, config, weights, experts, tokens)
            },
            RoutedSharedMoeWeights::NvFp4(expert_weights) => nvfp4::NvFp4RoutedExecution::new(
                backend, config, weights, expert_weights, tokens, phase,
            )
            .map(Box::new)
            .map(Self::NvFp4),
        }
    }

    pub(super) fn execute(
        &mut self,
        backend: &CudaBackend,
        weights: &AffineSharedExpertMoeWeights,
        input: &DeviceBuffer<bf16>,
        scratch: &mut AffineSharedMoeScratch,
    ) -> Result<()> {
        match (self, &weights.routed) {
            (Self::Affine { router, experts }, RoutedSharedMoeWeights::Affine(routed_weights)) => {
                let router_weight = match &weights.router {
                    crate::backend::linear::CheckpointProjectionWeight::Affine(weight) => {
                        weight.tensors()
                    },
                    _ => {
                        return Err(crate::Error::InvalidExecutionPlan(
                            "affine routed experts have a non-affine router",
                        ));
                    },
                };
                let selection = router.execute(input, router_weight)?;
                experts.execute(
                    input,
                    selection.indices,
                    selection.weights,
                    routed_weights,
                    &mut scratch.routed_intermediate,
                    &mut scratch.routed_output,
                )
            },
            (
                Self::Dense {
                    router,
                    top_k,
                    scores,
                    selected,
                    routing,
                    experts,
                    intermediate,
                },
                RoutedSharedMoeWeights::Dense(expert_weights),
            ) => {
                router.execute(input, scores)?;
                top_k.execute(backend.stream(), scores, selected, routing)?;
                experts.execute(
                    input,
                    selected,
                    routing,
                    expert_weights,
                    intermediate,
                    &mut scratch.routed_output,
                )
            },
            (
                Self::MxFp4 {
                    router,
                    top_k,
                    scores,
                    selected,
                    routing,
                    experts,
                },
                RoutedSharedMoeWeights::MxFp4(expert_weights),
            ) => {
                router.execute(input, scores)?;
                top_k.execute(backend.stream(), scores, selected, routing)?;
                experts.execute(
                    input,
                    selected,
                    routing,
                    expert_weights,
                    &mut scratch.routed_output,
                )
            },
            (
                Self::MxFp8 {
                    router,
                    top_k,
                    scores,
                    selected,
                    routing,
                    experts,
                },
                RoutedSharedMoeWeights::MxFp8(expert_weights),
            ) => {
                router.execute(input, scores)?;
                top_k.execute(backend.stream(), scores, selected, routing)?;
                experts.execute(
                    input,
                    selected,
                    routing,
                    expert_weights,
                    &mut scratch.routed_output,
                )
            },
            (Self::NvFp4(experts), RoutedSharedMoeWeights::NvFp4(expert_weights)) => {
                experts.execute(backend, input, expert_weights, &mut scratch.routed_output)
            },
            _ => Err(crate::Error::InvalidExecutionPlan(
                "shared-routed execution differs from checkpoint storage",
            )),
        }
    }
}
