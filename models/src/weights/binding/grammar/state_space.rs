//! Binding grammar of state-space hybrid layers: Mamba-2 or attention mixers
//! paired with an optional latent routed-expert block.

use super::{has_attention, has_layer, invalid};
use crate::{
    error::Result,
    semantic::{BlockLayoutSpec, DecoderLayerSpec, FeedForwardSpec, MixerSpec, SemanticModelSpec},
    weights::binding::{
        AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole,
        LatentProjectionRole, LayerTensorRole, LogicalTensorRole, StateSpaceTensorRole,
        TensorBinding, TensorStorage,
    },
};

const STATE_SPACE: [StateSpaceTensorRole; 7] = [
    StateSpaceTensorRole::DecayLog,
    StateSpaceTensorRole::SkipScale,
    StateSpaceTensorRole::Convolution,
    StateSpaceTensorRole::TimeBias,
    StateSpaceTensorRole::InputProjection,
    StateSpaceTensorRole::GatedNorm,
    StateSpaceTensorRole::OutputProjection,
];

pub(super) fn validate(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    let paired = matches!(spec.decoder.block_layout, BlockLayoutSpec::Paired { .. });
    if paired {
        unrecognized(tensors)?;
    }
    for layer in &spec.decoder.layers {
        let state_space = matches!(layer.mixer, MixerSpec::StateSpace(_));
        if !paired && !state_space {
            continue;
        }
        if !paired {
            return Err(invalid("state-space mixers require a paired block layout"));
        }
        if !mixer(layer, tensors)? || !feed_forward(layer, tensors) {
            return Err(invalid(format!(
                "semantic state-space hybrid layer {} has an incomplete binding grammar",
                layer.index
            )));
        }
    }
    Ok(())
}

fn mixer(layer: &DecoderLayerSpec, tensors: &[TensorBinding]) -> Result<bool> {
    let index = layer.index;
    if !has_layer(tensors, index, LayerTensorRole::InputNorm) {
        return Ok(false);
    }
    match &layer.mixer {
        MixerSpec::StateSpace(state) => {
            let complete = STATE_SPACE
                .into_iter()
                .all(|tensor| has_layer(tensors, index, LayerTensorRole::StateSpace { tensor }));
            let bias = convolution_bias(tensors, index);
            if complete && bias != state.convolution_bias {
                return Err(invalid(format!(
                    "semantic layer {index} convolution bias does not match the configuration"
                )));
            }
            Ok(complete)
        },
        MixerSpec::SoftmaxAttention(_) => Ok([
            AttentionProjectionRole::Query,
            AttentionProjectionRole::Key,
            AttentionProjectionRole::Value,
            AttentionProjectionRole::Output,
        ]
        .into_iter()
        .all(|projection| has_attention(tensors, index, projection))
            && !has_layer(tensors, index, LayerTensorRole::QueryNorm)
            && !has_layer(tensors, index, LayerTensorRole::KeyNorm)),
        MixerSpec::LinearAttention(_) => Ok(false),
    }
}

fn feed_forward(layer: &DecoderLayerSpec, tensors: &[TensorBinding]) -> bool {
    let index = layer.index;
    let has = |tensor| has_layer(tensors, index, tensor);
    match &layer.feed_forward {
        FeedForwardSpec::None => {
            !has(LayerTensorRole::PostAttentionNorm) && !has(LayerTensorRole::Router)
        },
        FeedForwardSpec::LatentRouted { routed, shared, .. } => {
            let experts = (0..routed.expert_count).all(|expert| {
                [ExpertProjectionRole::Up, ExpertProjectionRole::Down].into_iter().all(
                    |projection| {
                        has(LayerTensorRole::ExpertProjection { expert: Some(expert), projection })
                    },
                ) && !has(LayerTensorRole::ExpertProjection {
                    expert: Some(expert),
                    projection: ExpertProjectionRole::Gate,
                })
            });
            let shared_projection =
                |projection| has(LayerTensorRole::SharedExpertProjection { projection });
            let shared_complete = match shared {
                Some(_) => {
                    shared_projection(FeedForwardProjectionRole::Up)
                        && shared_projection(FeedForwardProjectionRole::Down)
                        && !shared_projection(FeedForwardProjectionRole::Gate)
                },
                None => !shared_projection(FeedForwardProjectionRole::Up),
            };
            experts
                && shared_complete
                && has(LayerTensorRole::PostAttentionNorm)
                && has(LayerTensorRole::Router)
                && has(LayerTensorRole::RouterCorrectionBias)
                && has(LayerTensorRole::LatentProjection {
                    projection: LatentProjectionRole::Input,
                })
                && has(LayerTensorRole::LatentProjection {
                    projection: LatentProjectionRole::Output,
                })
        },
        FeedForwardSpec::Dense { .. }
        | FeedForwardSpec::Routed { .. }
        | FeedForwardSpec::DenseAndRouted { .. } => false,
    }
}

fn convolution_bias(tensors: &[TensorBinding], index: usize) -> bool {
    let role = LogicalTensorRole::Layer {
        index,
        tensor: LayerTensorRole::StateSpace {
            tensor: StateSpaceTensorRole::Convolution,
        },
    };
    tensors
        .binary_search_by(|binding| binding.role.cmp(&role))
        .ok()
        .and_then(|position| tensors.get(position))
        .is_some_and(|binding| {
            matches!(binding.storage, TensorStorage::Dense { bias: Some(_), .. })
        })
}

/// Every tensor inside a paired checkpoint block must have a typed role in
/// the block it was found in; unknown or misplaced tensors are rejected.
fn unrecognized(tensors: &[TensorBinding]) -> Result<()> {
    let unknown = tensors.iter().find(|binding| match &binding.role {
        LogicalTensorRole::Layer {
            tensor: LayerTensorRole::Auxiliary { .. },
            ..
        } => true,
        LogicalTensorRole::Auxiliary { path } => path.starts_with("backbone.layers."),
        _ => false,
    });
    unknown.map_or(Ok(()), |binding| {
        Err(invalid(format!(
            "hybrid checkpoint tensor {} has no role in its block",
            binding.source
        )))
    })
}
