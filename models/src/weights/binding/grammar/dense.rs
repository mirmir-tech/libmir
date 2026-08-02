use super::{
    AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LayerTensorRole,
    TensorBinding, has_attention, has_expert, has_layer, individual_experts, invalid,
};
use crate::{
    error::Result,
    semantic::{FeedForwardSpec, KeyValueRelation, MixerSpec, SemanticModelSpec},
};

pub(super) fn validate(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    for layer in &spec.decoder.layers {
        let FeedForwardSpec::DenseAndRouted { routed, .. } = &layer.feed_forward else {
            continue;
        };
        let common = [
            LayerTensorRole::InputNorm,
            LayerTensorRole::QueryNorm,
            LayerTensorRole::KeyNorm,
            LayerTensorRole::PostAttentionNorm,
            LayerTensorRole::PreDenseNorm,
            LayerTensorRole::PostDenseNorm,
            LayerTensorRole::Router,
            LayerTensorRole::RouterNormScale,
            LayerTensorRole::RouterExpertScale,
            LayerTensorRole::PreExpertNorm,
            LayerTensorRole::PostExpertNorm,
            LayerTensorRole::PostFeedForwardNorm,
            LayerTensorRole::LayerScale,
        ]
        .into_iter()
        .all(|role| has_layer(tensors, layer.index, role));
        let attention = match &layer.mixer {
            MixerSpec::SoftmaxAttention(attention) => {
                [
                    AttentionProjectionRole::Query,
                    AttentionProjectionRole::Key,
                    AttentionProjectionRole::Output,
                ]
                .into_iter()
                .all(|projection| has_attention(tensors, layer.index, projection))
                    && (attention.key_value_relation == KeyValueRelation::KeyEqualsValue
                        || has_attention(tensors, layer.index, AttentionProjectionRole::Value))
            },
            MixerSpec::LinearAttention(_) => false,
        };
        let dense = [
            FeedForwardProjectionRole::Gate,
            FeedForwardProjectionRole::Up,
            FeedForwardProjectionRole::Down,
        ]
        .into_iter()
        .all(|projection| {
            has_layer(tensors, layer.index, LayerTensorRole::FeedForwardProjection { projection })
        });
        let stacked =
            [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down]
                .into_iter()
                .all(|projection| has_expert(tensors, layer.index, projection));
        let fused_stacked = has_expert(tensors, layer.index, ExpertProjectionRole::GateUp)
            && has_expert(tensors, layer.index, ExpertProjectionRole::Down);
        let individual = individual_experts(tensors, layer.index, routed.expert_count);
        if !common
            || !attention
            || !dense
            || usize::from(stacked) + usize::from(fused_stacked) + usize::from(individual) != 1
        {
            return Err(invalid(format!(
                "semantic dense-and-routed layer {} has an incomplete or ambiguous binding grammar",
                layer.index
            )));
        }
    }
    Ok(())
}
