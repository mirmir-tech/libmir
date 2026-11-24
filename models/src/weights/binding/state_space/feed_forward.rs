//! The feed-forward block of a state-space hybrid layer.

use super::{
    DenseHybridBindings, LatentProjectionBindings, LatentRoutedBindings,
    StateSpaceHybridFeedForwardBindings, UngatedExpertBindings, invalid, optional, required,
};
use crate::{
    error::Result,
    weights::{
        ExpertProjectionRole, FeedForwardProjectionRole, LatentProjectionRole, LayerTensorRole,
        TensorBinding, WeightBindingPlan,
        binding::{slot::LayerSlot, view::RoutedExpertBindings},
    },
};

/// Routed experts where the layer has a router, a dense MLP where it has a
/// feed-forward up projection, else none.
pub(super) fn bindings(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
) -> Result<Option<StateSpaceHybridFeedForwardBindings<'_>>> {
    if let Some(router) = optional(plan, slot, LayerTensorRole::Router) {
        return latent_routed(plan, slot, router)
            .map(|routed| Some(StateSpaceHybridFeedForwardBindings::Routed(routed)));
    }
    let projection = |projection| LayerTensorRole::FeedForwardProjection { projection };
    let Some(up) = optional(plan, slot, projection(FeedForwardProjectionRole::Up)) else {
        return Ok(None);
    };
    Ok(Some(StateSpaceHybridFeedForwardBindings::Dense(DenseHybridBindings {
        norm: required(plan, slot, LayerTensorRole::PostAttentionNorm)?,
        up,
        down: required(plan, slot, projection(FeedForwardProjectionRole::Down))?,
    })))
}

/// Every binding of a feed-forward block.
pub(super) fn sources(
    feed_forward: StateSpaceHybridFeedForwardBindings<'_>,
) -> Vec<&TensorBinding> {
    match feed_forward {
        StateSpaceHybridFeedForwardBindings::Dense(dense) => vec![dense.norm, dense.up, dense.down],
        StateSpaceHybridFeedForwardBindings::Routed(routed) => {
            let mut bindings = vec![routed.norm, routed.router, routed.correction_bias];
            if let Some(latent) = routed.latent {
                bindings.extend([latent.input, latent.output]);
            }
            bindings.extend(routed.routed.bindings());
            if let Some(shared) = routed.shared {
                bindings.extend([shared.up, shared.down]);
            }
            bindings
        },
    }
}

fn latent_routed<'a>(
    plan: &'a WeightBindingPlan,
    slot: LayerSlot,
    router: &'a TensorBinding,
) -> Result<LatentRoutedBindings<'a>> {
    let get = |tensor| required(plan, slot, tensor);
    let projection =
        |projection| optional(plan, slot, LayerTensorRole::LatentProjection { projection });
    let latent = match (
        projection(LatentProjectionRole::Input),
        projection(LatentProjectionRole::Output),
    ) {
        (Some(input), Some(output)) => Some(LatentProjectionBindings { input, output }),
        (None, None) => None,
        _ => return Err(invalid(slot, "latent space has only one projection")),
    };
    let stacked = |projection| {
        optional(plan, slot, LayerTensorRole::ExpertProjection { expert: None, projection })
    };
    let experts = match (stacked(ExpertProjectionRole::Up), stacked(ExpertProjectionRole::Down)) {
        (Some(up), Some(down)) => RoutedExpertBindings::UngatedStacked { up, down },
        (None, None) => RoutedExpertBindings::Individual { tensors: &plan.tensors, layer: slot },
        _ => return Err(invalid(slot, "stacked experts have only one projection")),
    };
    let shared =
        |projection| optional(plan, slot, LayerTensorRole::SharedExpertProjection { projection });
    let shared =
        match (shared(FeedForwardProjectionRole::Up), shared(FeedForwardProjectionRole::Down)) {
            (Some(up), Some(down)) => Some(UngatedExpertBindings { up, down }),
            (None, None) => None,
            _ => return Err(invalid(slot, "shared expert has only one projection")),
        };
    Ok(LatentRoutedBindings {
        norm: get(LayerTensorRole::PostAttentionNorm)?,
        router,
        correction_bias: get(LayerTensorRole::RouterCorrectionBias)?,
        latent,
        routed: experts,
        shared,
    })
}
