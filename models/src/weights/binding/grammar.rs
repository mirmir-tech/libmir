use super::{
    AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LayerTensorRole,
    LinearAttentionTensorRole, LogicalTensorRole, TensorBinding,
};
use crate::{
    error::{ModelsError, Result},
    semantic::{
        ActivationSpec, FeedForwardSpec, KeyValueRelation, MixerSpec, RoutedExpertsSpec,
        SemanticModelSpec,
    },
};

pub(super) fn validate(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    attention(spec, tensors)?;
    clamped_routed(spec, tensors)?;
    shared_routed(spec, tensors)?;
    dense_and_routed(spec, tensors)
}

fn attention(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    for layer in &spec.decoder.layers {
        let fused = has_attention(tensors, layer.index, AttentionProjectionRole::Qkv);
        let separate = [
            AttentionProjectionRole::Query,
            AttentionProjectionRole::Key,
            AttentionProjectionRole::Value,
        ]
        .into_iter()
        .all(|projection| has_attention(tensors, layer.index, projection));
        if fused && separate {
            return Err(invalid(format!(
                "semantic layer {} matches both fused and separate QKV binding grammars",
                layer.index
            )));
        }
    }
    Ok(())
}

fn has_attention(
    tensors: &[TensorBinding],
    index: usize,
    projection: AttentionProjectionRole,
) -> bool {
    tensors.iter().any(|binding| {
        binding.role
            == LogicalTensorRole::Layer {
                index,
                tensor: LayerTensorRole::AttentionProjection { projection },
            }
    })
}

fn clamped_routed(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    let layers = spec.decoder.layers.iter().filter(|layer| {
        matches!(
            &layer.feed_forward,
            FeedForwardSpec::Routed {
                routed: RoutedExpertsSpec {
                    activation: ActivationSpec::SwiGlu { clamp: Some(_), .. },
                    ..
                },
                shared: None,
            }
        )
    });
    let mut routed_layers = 0;
    let mut interleaved_complete = true;
    let mut separate_complete = true;
    for layer in layers {
        routed_layers += 1;
        interleaved_complete &= has_expert(tensors, layer.index, ExpertProjectionRole::GateUp)
            && has_expert(tensors, layer.index, ExpertProjectionRole::Down);
        separate_complete &= has_expert(tensors, layer.index, ExpertProjectionRole::Gate)
            && has_expert(tensors, layer.index, ExpertProjectionRole::Up)
            && has_expert(tensors, layer.index, ExpertProjectionRole::Down);
    }
    if routed_layers == 0 {
        return Ok(());
    }
    match (interleaved_complete, separate_complete) {
        (true, false) | (false, true) => Ok(()),
        (true, true) => Err(invalid(
            "checkpoint matches both interleaved and separate routed-expert binding grammars",
        )),
        (false, false) => Err(invalid("checkpoint has no complete routed-expert binding grammar")),
    }
}

fn has_expert(tensors: &[TensorBinding], index: usize, projection: ExpertProjectionRole) -> bool {
    tensors.iter().any(|binding| {
        binding.role
            == LogicalTensorRole::Layer {
                index,
                tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
            }
    })
}

fn shared_routed(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    for layer in &spec.decoder.layers {
        let FeedForwardSpec::Routed { shared: Some(_), .. } = &layer.feed_forward else {
            continue;
        };
        let routed =
            [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down]
                .into_iter()
                .all(|projection| has_expert(tensors, layer.index, projection));
        let shared = [
            FeedForwardProjectionRole::Gate,
            FeedForwardProjectionRole::Up,
            FeedForwardProjectionRole::Down,
        ]
        .into_iter()
        .all(|projection| {
            has_layer(tensors, layer.index, LayerTensorRole::SharedExpertProjection { projection })
        });
        let complete = routed
            && shared
            && has_layer(tensors, layer.index, LayerTensorRole::Router)
            && has_layer(tensors, layer.index, LayerTensorRole::SharedExpertOutputGate)
            && mixer_complete(&layer.mixer, tensors, layer.index);
        let interleaved = has_expert(tensors, layer.index, ExpertProjectionRole::GateUp);
        if !complete || interleaved {
            return Err(invalid(format!(
                "semantic shared-expert layer {} has an incomplete or ambiguous binding grammar",
                layer.index
            )));
        }
    }
    Ok(())
}

fn dense_and_routed(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
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
        let individual = (0..routed.expert_count).all(|expert| {
            [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down]
                .into_iter()
                .all(|projection| {
                    has_layer(
                        tensors,
                        layer.index,
                        LayerTensorRole::ExpertProjection { expert: Some(expert), projection },
                    )
                })
        });
        if !common || !attention || !dense || stacked == individual {
            return Err(invalid(format!(
                "semantic dense-and-routed layer {} has an incomplete or ambiguous binding grammar",
                layer.index
            )));
        }
    }
    Ok(())
}

fn mixer_complete(mixer: &MixerSpec, tensors: &[TensorBinding], index: usize) -> bool {
    match mixer {
        MixerSpec::LinearAttention(_) => [
            LinearAttentionTensorRole::DecayLog,
            LinearAttentionTensorRole::Convolution,
            LinearAttentionTensorRole::TimeBias,
            LinearAttentionTensorRole::QkvProjection,
            LinearAttentionTensorRole::GateProjection,
            LinearAttentionTensorRole::AlphaProjection,
            LinearAttentionTensorRole::BetaProjection,
            LinearAttentionTensorRole::Norm,
            LinearAttentionTensorRole::OutputProjection,
        ]
        .into_iter()
        .all(|tensor| has_layer(tensors, index, LayerTensorRole::LinearAttention { tensor })),
        MixerSpec::SoftmaxAttention(_) => {
            [
                AttentionProjectionRole::Query,
                AttentionProjectionRole::Key,
                AttentionProjectionRole::Value,
                AttentionProjectionRole::Output,
            ]
            .into_iter()
            .all(|projection| has_attention(tensors, index, projection))
                && has_layer(tensors, index, LayerTensorRole::QueryNorm)
                && has_layer(tensors, index, LayerTensorRole::KeyNorm)
        },
    }
}

fn has_layer(tensors: &[TensorBinding], index: usize, tensor: LayerTensorRole) -> bool {
    let role = LogicalTensorRole::Layer { index, tensor };
    tensors.iter().any(|binding| binding.role == role)
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
