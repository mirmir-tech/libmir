use super::{
    AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LayerTensorRole,
    LinearAttentionTensorRole, LogicalTensorRole, TensorBinding,
};
use crate::{
    error::{ModelsError, Result},
    semantic::{ActivationSpec, FeedForwardSpec, MixerSpec, RoutedExpertsSpec, SemanticModelSpec},
};

mod dense;

pub(super) fn validate(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    attention(spec, tensors)?;
    clamped_routed(spec, tensors)?;
    shared_routed(spec, tensors)?;
    dense::validate(spec, tensors)
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
    has_layer(tensors, index, LayerTensorRole::AttentionProjection { projection })
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
    has_layer(tensors, index, LayerTensorRole::ExpertProjection { expert: None, projection })
}

fn shared_routed(spec: &SemanticModelSpec, tensors: &[TensorBinding]) -> Result<()> {
    for layer in &spec.decoder.layers {
        let FeedForwardSpec::Routed { routed, shared: Some(_) } = &layer.feed_forward else {
            continue;
        };
        let separate =
            [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down]
                .into_iter()
                .all(|projection| has_expert(tensors, layer.index, projection));
        let fused = has_expert(tensors, layer.index, ExpertProjectionRole::GateUp)
            && has_expert(tensors, layer.index, ExpertProjectionRole::Down);
        let shared = [
            FeedForwardProjectionRole::Gate,
            FeedForwardProjectionRole::Up,
            FeedForwardProjectionRole::Down,
        ]
        .into_iter()
        .all(|projection| {
            has_layer(tensors, layer.index, LayerTensorRole::SharedExpertProjection { projection })
        });
        let individual = individual_experts(tensors, layer.index, routed.expert_count);
        let complete = usize::from(separate) + usize::from(fused) + usize::from(individual) == 1
            && shared
            && has_layer(tensors, layer.index, LayerTensorRole::Router)
            && has_layer(tensors, layer.index, LayerTensorRole::SharedExpertOutputGate)
            && mixer_complete(&layer.mixer, tensors, layer.index);
        if !complete {
            return Err(invalid(format!(
                "semantic shared-expert layer {} has an incomplete or ambiguous binding grammar",
                layer.index
            )));
        }
    }
    Ok(())
}

fn individual_experts(tensors: &[TensorBinding], index: usize, count: usize) -> bool {
    (0..count).all(|expert| {
        [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down]
            .into_iter()
            .all(|projection| {
                has_layer(
                    tensors,
                    index,
                    LayerTensorRole::ExpertProjection { expert: Some(expert), projection },
                )
            })
    })
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
    tensors.binary_search_by(|binding| binding.role.cmp(&role)).is_ok()
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
