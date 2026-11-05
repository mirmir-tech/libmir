use super::{SharedRoutedFeedForwardBindings, WeightBindingPlan, invalid, layer, optional};
use crate::{
    error::Result,
    weights::{
        ExpertProjectionRole, FeedForwardProjectionRole, LayerTensorRole,
        binding::{slot::LayerSlot, view::RoutedExpertBindings},
    },
};

pub(super) fn feed_forward(
    plan: &WeightBindingPlan,
    index: usize,
) -> Result<SharedRoutedFeedForwardBindings<'_>> {
    let expert = |projection| {
        layer(plan, index, LayerTensorRole::ExpertProjection { expert: None, projection })
    };
    let optional_expert = |projection| {
        optional(plan, index, LayerTensorRole::ExpertProjection { expert: None, projection })
    };
    let shared =
        |projection| layer(plan, index, LayerTensorRole::SharedExpertProjection { projection });
    Ok(SharedRoutedFeedForwardBindings {
        router: layer(plan, index, LayerTensorRole::Router)?,
        routed: if let Some(gate_up) = optional_expert(ExpertProjectionRole::GateUp) {
            RoutedExpertBindings::InterleavedGateUp {
                gate_up,
                down: expert(ExpertProjectionRole::Down)?,
            }
        } else if optional_expert(ExpertProjectionRole::Down).is_some() {
            RoutedExpertBindings::SeparateGateUp {
                gate: expert(ExpertProjectionRole::Gate)?,
                up: expert(ExpertProjectionRole::Up)?,
                down: expert(ExpertProjectionRole::Down)?,
            }
        } else {
            let routed = RoutedExpertBindings::Individual {
                tensors: &plan.tensors,
                layer: LayerSlot::Decoder(index),
            };
            let gate = routed.individual(ExpertProjectionRole::Gate);
            let up = routed.individual(ExpertProjectionRole::Up);
            let down = routed.individual(ExpertProjectionRole::Down);
            if gate.is_empty() || gate.len() != up.len() || gate.len() != down.len() {
                return Err(invalid(index, "incomplete individual expert projection set"));
            }
            routed
        },
        shared_gate: shared(FeedForwardProjectionRole::Gate)?,
        shared_up: shared(FeedForwardProjectionRole::Up)?,
        shared_down: shared(FeedForwardProjectionRole::Down)?,
        shared_output_gate: layer(plan, index, LayerTensorRole::SharedExpertOutputGate)?,
    })
}
