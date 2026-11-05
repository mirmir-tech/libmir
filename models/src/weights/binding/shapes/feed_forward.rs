use crate::semantic::{FeedForwardSpec, RoutedExpertsSpec, SharedExpertSpec};

pub(super) fn routed(feed_forward: &FeedForwardSpec) -> Option<&RoutedExpertsSpec> {
    match feed_forward {
        FeedForwardSpec::Routed { routed, .. }
        | FeedForwardSpec::DenseAndRouted { routed, .. }
        | FeedForwardSpec::LatentRouted { routed, .. } => Some(routed),
        FeedForwardSpec::Dense { .. } | FeedForwardSpec::None => None,
    }
}

pub(super) fn shared(feed_forward: &FeedForwardSpec) -> Option<&SharedExpertSpec> {
    match feed_forward {
        FeedForwardSpec::Routed { shared: Some(shared), .. }
        | FeedForwardSpec::LatentRouted { shared: Some(shared), .. } => Some(shared),
        FeedForwardSpec::Dense { .. }
        | FeedForwardSpec::Routed { shared: None, .. }
        | FeedForwardSpec::LatentRouted { shared: None, .. }
        | FeedForwardSpec::DenseAndRouted { .. }
        | FeedForwardSpec::None => None,
    }
}

/// Width routed experts read and write: the latent size for latent experts,
/// otherwise the hidden size.
pub(super) const fn expert_width(feed_forward: &FeedForwardSpec, hidden: usize) -> usize {
    match feed_forward {
        FeedForwardSpec::LatentRouted { latent_size, .. } => *latent_size,
        FeedForwardSpec::Dense { .. }
        | FeedForwardSpec::Routed { .. }
        | FeedForwardSpec::DenseAndRouted { .. }
        | FeedForwardSpec::None => hidden,
    }
}
