//! Logical shapes of state-space mixer and latent routed-expert tensors.

use super::{LatentProjectionRole, StateSpaceTensorRole, feed_forward::routed};
use crate::semantic::{FeedForwardSpec, MixerSpec, RouterNormalization};

pub(super) fn mixer(
    mixer: &MixerSpec,
    hidden: usize,
    tensor: StateSpaceTensorRole,
) -> Option<Vec<usize>> {
    let MixerSpec::StateSpace(state) = mixer else {
        return None;
    };
    let inner = state.inner_size();
    Some(match tensor {
        StateSpaceTensorRole::DecayLog
        | StateSpaceTensorRole::SkipScale
        | StateSpaceTensorRole::TimeBias => vec![state.heads],
        // Stored as `[channels, 1, kernel]`; validation accepts that layout.
        StateSpaceTensorRole::Convolution => {
            vec![state.convolution_channels(), state.convolution_kernel_size]
        },
        StateSpaceTensorRole::InputProjection => vec![state.input_projection_size(), hidden],
        StateSpaceTensorRole::GatedNorm => vec![inner],
        StateSpaceTensorRole::OutputProjection => vec![hidden, inner],
    })
}

pub(super) fn latent(
    feed_forward: &FeedForwardSpec,
    hidden: usize,
    projection: LatentProjectionRole,
) -> Option<Vec<usize>> {
    let FeedForwardSpec::LatentRouted { latent_size: Some(latent), .. } = feed_forward else {
        return None;
    };
    Some(match projection {
        LatentProjectionRole::Input => vec![*latent, hidden],
        LatentProjectionRole::Output => vec![hidden, *latent],
    })
}

/// One bias per expert for latent experts and for sigmoid routers that
/// select by a corrected score.
pub(super) fn correction_bias(feed_forward: &FeedForwardSpec) -> Option<Vec<usize>> {
    let routed = routed(feed_forward)?;
    (matches!(feed_forward, FeedForwardSpec::LatentRouted { .. })
        || routed.router_normalization == RouterNormalization::SigmoidBiasedTopK)
        .then(|| vec![routed.expert_count])
}
