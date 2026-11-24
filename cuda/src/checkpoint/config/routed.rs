//! The feed-forward of a dense-decoder layer as the semantic spec states it.

use architecture::{
    cuda::{CudaDecoderPlan, CudaLayer},
    lowering::{Experts, FeedForwardLowering},
};
use models::semantic::{FeedForwardSpec, SemanticModelSpec, SharedExpertSpec};

use super::unsupported;
use crate::Result;

pub(super) fn feed_forward(semantic: &SemanticModelSpec, layer: usize) -> Result<&FeedForwardSpec> {
    semantic
        .decoder
        .layers
        .get(layer)
        .map(|spec| &spec.feed_forward)
        .ok_or_else(|| unsupported("layer is missing from the model spec"))
}

/// The layer as admission lowered it.
pub(super) fn lowered(plan: &CudaDecoderPlan, layer: usize) -> Result<&CudaLayer> {
    plan.layers()
        .get(layer)
        .ok_or_else(|| unsupported("layer is missing from the admitted plan"))
}

/// The routed experts of an admitted layer.
pub(super) fn experts(layer: &CudaLayer) -> Result<Experts> {
    match layer.feed_forward {
        FeedForwardLowering::Routed { experts, .. }
        | FeedForwardLowering::DenseAndRouted { experts, .. }
        | FeedForwardLowering::LatentRouted { experts, .. } => Ok(experts),
        FeedForwardLowering::Dense { .. } | FeedForwardLowering::None => {
            Err(unsupported("a decode MoE layer has no routed experts"))
        },
    }
}

/// The router of a dense layer's routed experts, which select from every
/// expert, and the ungated shared expert beside them.
pub(super) fn dense_router(
    lowered: &Experts,
    shared: Option<&SharedExpertSpec>,
    experts: crate::RoutedExpertsConfig,
) -> Result<crate::DenseRoutedExperts> {
    let scoring = crate::kernels::RouterUnitScoring::of_experts(lowered)?;
    if scoring.group_limit.is_some() {
        return Err(unsupported("a group-limited router in a dense layer"));
    }
    if shared.is_some_and(|shared| shared.gated_output) {
        return Err(unsupported("a gated shared expert in a dense layer"));
    }
    Ok(crate::DenseRoutedExperts {
        experts,
        weighting: scoring.weighting,
        scale: scoring.scale,
        shared_intermediate: shared
            .map(|shared| {
                u32::try_from(shared.intermediate_size)
                    .ok()
                    .and_then(std::num::NonZeroU32::new)
                    .ok_or_else(|| unsupported("an empty or oversized shared expert"))
            })
            .transpose()?,
    })
}
