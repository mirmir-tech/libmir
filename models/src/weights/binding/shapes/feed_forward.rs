use crate::semantic::{FeedForwardSpec, RoutedExpertsSpec, SharedExpertSpec};

pub(super) fn routed(feed_forward: &FeedForwardSpec) -> Option<&RoutedExpertsSpec> {
    match feed_forward {
        FeedForwardSpec::Routed { routed, .. } | FeedForwardSpec::DenseAndRouted { routed, .. } => {
            Some(routed)
        },
        FeedForwardSpec::Dense { .. } => None,
    }
}

pub(super) fn shared(feed_forward: &FeedForwardSpec) -> Option<&SharedExpertSpec> {
    match feed_forward {
        FeedForwardSpec::Routed { shared: Some(shared), .. } => Some(shared),
        FeedForwardSpec::Dense { .. }
        | FeedForwardSpec::Routed { shared: None, .. }
        | FeedForwardSpec::DenseAndRouted { .. } => None,
    }
}
