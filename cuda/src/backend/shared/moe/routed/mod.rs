use mircuda::{DeviceBuffer, bf16};

use self::affine::AutoAffineRoutedExperts;
use super::{
    AffineSharedExpertMoeConfig,
    scratch::AffineSharedMoeScratch,
    weights::{AffineSharedExpertMoeWeights, RoutedSharedMoeWeights},
};
use crate::{
    AffineQuantizedConfig, AffineRouterBf16, CudaBackend, ExecutionPhase, Result,
    backend::{linear::SelectedDenseMoeBf16, router::BoundRouter},
};

mod affine;
mod ggml;
mod mxfp;
mod nvfp4;
mod parallel;
mod prequant;

#[derive(Debug)]
pub(super) enum SharedRoutedExecution {
    Affine {
        router: Box<AffineRouterBf16>,
        experts: Box<AutoAffineRoutedExperts>,
    },
    Dense {
        router: BoundRouter,
        experts: Box<SelectedDenseMoeBf16>,
        intermediate: DeviceBuffer<bf16>,
    },
    MxFp4 {
        router: BoundRouter,
        experts: Box<crate::backend::linear::MxFp4GatheredMoeBf16>,
    },
    MxFp8 {
        router: BoundRouter,
        experts: Box<crate::backend::linear::MxFp8GatheredMoeBf16>,
    },
    NvFp4(Box<nvfp4::NvFp4RoutedExecution>),
    Ggml(Box<ggml::GgmlRoutedExecution>),
}

impl SharedRoutedExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        weights: &AffineSharedExpertMoeWeights,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let bias = weights.router_correction_bias.as_ref();
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
                    config.routing,
                    bias,
                )?),
                experts: Box::new(AutoAffineRoutedExperts::new(backend, config, weights, tokens)?),
            }),
            RoutedSharedMoeWeights::Dense(experts) => Ok(Self::Dense {
                router: mxfp::router(backend, config, weights, tokens)?,
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
            }),
            RoutedSharedMoeWeights::MxFp4(experts) => {
                mxfp::mxfp4(backend, config, weights, experts, tokens)
            },
            RoutedSharedMoeWeights::MxFp8(experts) => {
                mxfp::mxfp8(backend, config, weights, experts, tokens)
            },
            RoutedSharedMoeWeights::Ggml(_) => {
                ggml::GgmlRoutedExecution::new(backend, config, weights, tokens)
                    .map(Box::new)
                    .map(Self::Ggml)
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
                Self::Dense { router, experts, intermediate },
                RoutedSharedMoeWeights::Dense(expert_weights),
            ) => {
                let selection = router.route(backend, input)?;
                experts.execute(
                    input,
                    selection.indices,
                    selection.weights,
                    expert_weights,
                    intermediate,
                    &mut scratch.routed_output,
                )
            },
            (Self::MxFp4 { router, experts }, RoutedSharedMoeWeights::MxFp4(expert_weights)) => {
                let selection = router.route(backend, input)?;
                experts.execute(
                    input,
                    selection.indices,
                    selection.weights,
                    expert_weights,
                    &mut scratch.routed_output,
                )
            },
            (Self::MxFp8 { router, experts }, RoutedSharedMoeWeights::MxFp8(expert_weights)) => {
                let selection = router.route(backend, input)?;
                experts.execute(
                    input,
                    selection.indices,
                    selection.weights,
                    expert_weights,
                    &mut scratch.routed_output,
                )
            },
            (Self::NvFp4(experts), RoutedSharedMoeWeights::NvFp4(expert_weights)) => {
                experts.execute(backend, input, expert_weights, &mut scratch.routed_output)
            },
            (Self::Ggml(experts), RoutedSharedMoeWeights::Ggml(expert_weights)) => {
                experts.execute(backend, input, expert_weights, &mut scratch.routed_output)
            },
            _ => Err(crate::Error::InvalidExecutionPlan(
                "shared-routed execution differs from checkpoint storage",
            )),
        }
    }
}
