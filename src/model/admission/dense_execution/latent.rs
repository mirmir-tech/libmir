//! Dense (BF16) experts of multi-head latent attention decoders, which CUDA
//! routes through its shared-routed stack.

use models::{
    execution::DecoderExecutionContract,
    semantic::{ActivationSpec, FeedForwardSpec, MixerSpec, RouterNormalization},
    weights::{
        ExpertProjectionRole, MixedFeedForwardBindings, RoutedExpertBindings, TensorStorage,
    },
};

/// Every layer is latent attention with a dense feed-forward, or softmax-routed
/// `SwiGLU` experts plus shared experts whose projections are all dense.
pub(super) fn supported(contract: &DecoderExecutionContract) -> bool {
    contract.semantic.decoder.layers.iter().all(|layer| {
        if !matches!(layer.mixer, MixerSpec::LatentAttention(_)) {
            return false;
        }
        match &layer.feed_forward {
            FeedForwardSpec::Dense { .. } => true,
            FeedForwardSpec::Routed { routed, shared: Some(shared) } => {
                matches!(routed.activation, ActivationSpec::SwiGlu { clamp: None, .. })
                    && matches!(shared.activation, ActivationSpec::SwiGlu { clamp: None, .. })
                    && matches!(
                        routed.router_normalization,
                        RouterNormalization::SoftmaxTopK
                            | RouterNormalization::SoftmaxProbabilityTopK
                            | RouterNormalization::SigmoidBiasedTopK
                    )
                    && contract.bindings.latent_decoder_layer(layer.index).is_ok_and(|bindings| {
                        matches!(
                            bindings.feed_forward,
                            MixedFeedForwardBindings::SharedRouted(routed) if dense(routed.routed)
                        )
                    })
            },
            _ => false,
        }
    })
}

fn dense(bindings: RoutedExpertBindings<'_>) -> bool {
    let projections = match bindings {
        RoutedExpertBindings::InterleavedGateUp { gate_up, down } => vec![gate_up, down],
        RoutedExpertBindings::SeparateGateUp { gate, up, down } => vec![gate, up, down],
        RoutedExpertBindings::Individual { .. } => {
            let roles =
                [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down];
            if roles.iter().any(|role| bindings.individual(*role).is_empty()) {
                return false;
            }
            roles.into_iter().flat_map(|role| bindings.individual(role)).collect()
        },
        RoutedExpertBindings::UngatedStacked { .. } => return false,
    };
    projections
        .iter()
        .all(|binding| matches!(binding.storage, TensorStorage::Dense { .. }))
}
