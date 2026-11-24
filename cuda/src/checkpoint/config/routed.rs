//! The feed-forward of a dense-decoder layer as the semantic spec states it.

use models::semantic::{FeedForwardSpec, RoutedExpertsSpec, SemanticModelSpec, SharedExpertSpec};

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

/// The router of a dense layer's routed experts, which select from every
/// expert, and the ungated shared expert beside them.
pub(super) fn dense_router(
    spec: &RoutedExpertsSpec,
    shared: Option<&SharedExpertSpec>,
    experts: crate::RoutedExpertsConfig,
) -> Result<crate::DenseRoutedExperts> {
    let scoring = crate::kernels::RouterUnitScoring::of(spec)?;
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
