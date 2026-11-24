//! Completeness of multi-head latent attention bindings.

use super::{has_attention, has_layer, invalid};
use crate::{
    error::Result,
    semantic::{MixerSpec, RouterNormalization, SemanticModelSpec},
    weights::binding::{
        AttentionProjectionRole, LatentAttentionTensorRole, LayerTensorRole, TensorBinding,
    },
};

/// Every latent layer binds its key/value latent path, the output projection
/// and exactly one query path: the normalized query latent when the spec has
/// a query rank, the direct projection otherwise. Sigmoid routers bind their
/// correction bias.
pub(super) fn validate(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    for layer in &spec.decoder.layers {
        if !matches!(layer.mixer, MixerSpec::LatentAttention(_)) {
            continue;
        }
        if !complete(&layer.mixer, tensors, layer.index) {
            return Err(invalid(format!(
                "semantic latent-attention layer {} has an incomplete binding grammar",
                layer.index
            )));
        }
        let sigmoid = matches!(
            &layer.feed_forward,
            crate::semantic::FeedForwardSpec::Routed { routed, .. }
                if routed.router_normalization == RouterNormalization::SigmoidBiasedTopK
        );
        if sigmoid && !has_layer(tensors, layer.index, LayerTensorRole::RouterCorrectionBias) {
            return Err(invalid(format!(
                "semantic layer {} routes by sigmoid without its correction bias",
                layer.index
            )));
        }
    }
    Ok(())
}

pub(super) fn complete(mixer: &MixerSpec, tensors: &[TensorBinding], index: usize) -> bool {
    let MixerSpec::LatentAttention(latent) = mixer else {
        return false;
    };
    let has = |tensor| has_layer(tensors, index, LayerTensorRole::LatentAttention { tensor });
    // The up-projection is `kv_b_proj` or its per-head key and value halves.
    let fused = has(LatentAttentionTensorRole::KeyValueUp);
    let split = [LatentAttentionTensorRole::KeyUp, LatentAttentionTensorRole::ValueUp];
    let up = if fused {
        !split.into_iter().any(has)
    } else {
        split.into_iter().all(has)
    };
    let key_value = up
        && [LatentAttentionTensorRole::KeyValueDown, LatentAttentionTensorRole::KeyValueLatentNorm]
            .into_iter()
            .all(has);
    let direct = has_attention(tensors, index, AttentionProjectionRole::Query);
    let latent_query = [
        LatentAttentionTensorRole::QueryDown,
        LatentAttentionTensorRole::QueryLatentNorm,
        LatentAttentionTensorRole::QueryUp,
    ];
    let query = if latent.query_rank.is_some() {
        !direct && latent_query.into_iter().all(has)
    } else {
        direct && !latent_query.into_iter().any(has)
    };
    key_value && query && has_attention(tensors, index, AttentionProjectionRole::Output)
}
