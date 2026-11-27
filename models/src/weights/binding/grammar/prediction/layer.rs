//! Binding grammar of a prediction module that is one decoder layer (Qwen3.5's
//! `mtp.layers.N`): attention projections with learned Q/K norms where the
//! spec declares them, and a dense or shared-routed gated feed-forward.

use crate::{
    semantic::{DecoderLayerSpec, FeedForwardSpec, MixerSpec, QkNormalization},
    weights::binding::{
        AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LayerTensorRole,
    },
};

pub(super) fn complete(module: &DecoderLayerSpec, has: &dyn Fn(LayerTensorRole) -> bool) -> bool {
    let MixerSpec::SoftmaxAttention(attention) = &module.mixer else {
        return false;
    };
    let learned = matches!(
        attention.qk_normalization,
        QkNormalization::QueryKeyRms | QkNormalization::QueryKeyRmsAcrossHeads
    );
    let projections = [
        AttentionProjectionRole::Query,
        AttentionProjectionRole::Key,
        AttentionProjectionRole::Value,
        AttentionProjectionRole::Output,
    ]
    .into_iter()
    .all(|projection| has(LayerTensorRole::AttentionProjection { projection }));
    let norms =
        has(LayerTensorRole::QueryNorm) == learned && has(LayerTensorRole::KeyNorm) == learned;
    let gated = |role: fn(FeedForwardProjectionRole) -> LayerTensorRole| {
        [
            FeedForwardProjectionRole::Gate,
            FeedForwardProjectionRole::Up,
            FeedForwardProjectionRole::Down,
        ]
        .into_iter()
        .all(|projection| has(role(projection)))
    };
    let feed_forward = match &module.feed_forward {
        FeedForwardSpec::Dense { .. } => {
            gated(|projection| LayerTensorRole::FeedForwardProjection { projection })
        },
        FeedForwardSpec::Routed { shared: Some(shared), .. } => {
            let expert =
                |projection| has(LayerTensorRole::ExpertProjection { expert: None, projection });
            let experts = (expert(ExpertProjectionRole::Gate) && expert(ExpertProjectionRole::Up)
                || expert(ExpertProjectionRole::GateUp))
                && expert(ExpertProjectionRole::Down);
            has(LayerTensorRole::Router)
                && experts
                && gated(|projection| LayerTensorRole::SharedExpertProjection { projection })
                && has(LayerTensorRole::SharedExpertOutputGate) == shared.gated_output
        },
        _ => false,
    };
    has(LayerTensorRole::InputNorm)
        && has(LayerTensorRole::PostAttentionNorm)
        && projections
        && norms
        && feed_forward
}
