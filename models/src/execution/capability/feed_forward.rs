use std::collections::BTreeSet;

use super::ArchitectureCapability;
use crate::semantic::{ActivationSpec, FeedForwardSpec, RouterNormalization};

pub(super) fn feed_forward(
    feed_forward: &FeedForwardSpec,
    capabilities: &mut BTreeSet<ArchitectureCapability>,
) {
    match feed_forward {
        FeedForwardSpec::Dense { activation, .. } => {
            capabilities.insert(ArchitectureCapability::DenseFeedForward);
            activation_capability(activation, capabilities);
        },
        FeedForwardSpec::Routed { routed, shared } => {
            routed_capabilities(routed, capabilities);
            if let Some(shared) = shared {
                capabilities.insert(ArchitectureCapability::SharedExpert);
                activation_capability(&shared.activation, capabilities);
            }
        },
        FeedForwardSpec::DenseAndRouted { dense_activation, routed, .. } => {
            capabilities.insert(ArchitectureCapability::DenseFeedForward);
            capabilities.insert(ArchitectureCapability::DenseAndRouted);
            activation_capability(dense_activation, capabilities);
            routed_capabilities(routed, capabilities);
        },
        FeedForwardSpec::LatentRouted { routed, shared, .. } => {
            capabilities.insert(ArchitectureCapability::LatentRoutedExperts);
            routed_capabilities(routed, capabilities);
            if let Some(shared) = shared {
                capabilities.insert(ArchitectureCapability::SharedExpert);
                activation_capability(&shared.activation, capabilities);
            }
        },
        FeedForwardSpec::None => {},
    }
}

fn routed_capabilities(
    routed: &crate::semantic::RoutedExpertsSpec,
    capabilities: &mut BTreeSet<ArchitectureCapability>,
) {
    capabilities.insert(ArchitectureCapability::RoutedExperts);
    if (routed.output_scale - 1.0).abs() > f64::EPSILON {
        capabilities.insert(ArchitectureCapability::ScaledRoutedOutput);
    }
    capabilities.insert(match routed.router_normalization {
        RouterNormalization::SoftmaxTopK => ArchitectureCapability::SoftmaxTopKRouter,
        RouterNormalization::SoftmaxProbabilityTopK => {
            ArchitectureCapability::SoftmaxProbabilityTopKRouter
        },
        RouterNormalization::UnitTopK => ArchitectureCapability::UnitTopKRouter,
        RouterNormalization::SigmoidBiasedTopK => ArchitectureCapability::SigmoidBiasedTopKRouter,
    });
    activation_capability(&routed.activation, capabilities);
}

fn activation_capability(
    activation: &ActivationSpec,
    capabilities: &mut BTreeSet<ArchitectureCapability>,
) {
    capabilities.insert(match activation {
        ActivationSpec::SwiGlu { clamp: Some(_), .. } => ArchitectureCapability::ClampedSwiGlu,
        ActivationSpec::SwiGlu { clamp: None, .. } => ArchitectureCapability::SwiGlu,
        ActivationSpec::GeluTanh => ArchitectureCapability::GeluTanh,
        ActivationSpec::NamedGated { .. } => ArchitectureCapability::NamedGatedActivation,
        ActivationSpec::ReluSquared => ArchitectureCapability::ReluSquared,
    });
}
