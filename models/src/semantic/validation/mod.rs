use super::{
    ActivationSpec, FeedForwardSpec, MixerSpec, PositionEncodingSpec, SemanticModelSpec,
    model::CURRENT_SCHEMA_VERSION,
};
use crate::error::{ModelsError, Result};

mod sharing;

pub(super) fn validate(spec: &SemanticModelSpec) -> Result<()> {
    if spec.schema_version != CURRENT_SCHEMA_VERSION {
        return Err(invalid(format!(
            "unsupported semantic model schema version {}",
            spec.schema_version
        )));
    }
    let decoder = &spec.decoder;
    if decoder.hidden_size == 0 || decoder.vocab_size == 0 || decoder.layers.is_empty() {
        return Err(invalid("decoder dimensions and layer list must be non-empty"));
    }
    if [decoder.embedding_multiplier, decoder.residual_multiplier, decoder.logit_divisor]
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(invalid("embedding, residual and logit scales must be finite and positive"));
    }
    norm(decoder.final_norm.epsilon)?;
    layers(&decoder.layers, "decoder")?;
    layers(&decoder.prediction_layers, "prediction")?;
    if decoder.prediction_layers.iter().any(|layer| {
        !matches!(layer.mixer, MixerSpec::SoftmaxAttention(_))
            || layer.feed_forward == FeedForwardSpec::None
    }) {
        return Err(invalid("prediction layers must pair attention with a feed-forward"));
    }
    sharing::validate(decoder)?;
    block_layout(decoder)
}

fn layers(layers: &[super::DecoderLayerSpec], kind: &str) -> Result<()> {
    for (expected, layer) in layers.iter().enumerate() {
        if layer.index != expected {
            return Err(invalid(format!("{kind} layer indices must be contiguous and ordered")));
        }
        norm(layer.input_norm.epsilon)?;
        norm(layer.post_attention_norm.epsilon)?;
        mixer(&layer.mixer)?;
        feed_forward(&layer.feed_forward)?;
    }
    Ok(())
}

/// A paired layout must cover every checkpoint block exactly once: each
/// layer's mixer block directly follows the previous layer's last block.
fn block_layout(decoder: &super::DecoderSpec) -> Result<()> {
    let super::BlockLayoutSpec::Paired { physical_blocks, mixer_blocks } = &decoder.block_layout
    else {
        return Ok(());
    };
    if mixer_blocks.len() != decoder.layers.len() {
        return Err(invalid("paired block layout must name one mixer block per layer"));
    }
    let mut next = 0;
    for (mixer, layer) in mixer_blocks.iter().zip(&decoder.layers) {
        if *mixer != next {
            return Err(invalid("paired block layout must cover blocks contiguously"));
        }
        next = mixer
            + if layer.feed_forward == FeedForwardSpec::None {
                1
            } else {
                2
            };
    }
    if next != *physical_blocks {
        return Err(invalid("paired block layout does not cover every checkpoint block"));
    }
    Ok(())
}

fn norm(epsilon: f64) -> Result<()> {
    if epsilon.is_finite() && epsilon > 0.0 {
        Ok(())
    } else {
        Err(invalid("normalization epsilon must be finite and positive"))
    }
}

fn mixer(mixer: &MixerSpec) -> Result<()> {
    match mixer {
        MixerSpec::SoftmaxAttention(attention) => {
            if attention.query_heads == 0
                || attention.key_value_heads == 0
                || attention.head_dim == 0
                || !attention.query_heads.is_multiple_of(attention.key_value_heads)
                || !attention.scale.is_finite()
                || attention.scale <= 0.0
                || attention.query_temperature.is_some_and(|temperature| !temperature.is_valid())
            {
                return Err(invalid("invalid softmax attention specification"));
            }
            rotary(&attention.position)?;
        },
        MixerSpec::LinearAttention(linear) => {
            let dimensions = [
                linear.convolution_kernel_size,
                linear.key_heads,
                linear.value_heads,
                linear.key_head_dim,
                linear.value_head_dim,
            ];
            if dimensions.contains(&0) {
                return Err(invalid("invalid linear attention specification"));
            }
        },
        MixerSpec::StateSpace(state) => state_space(state)?,
        MixerSpec::LatentAttention(latent) => {
            let dimensions = [
                latent.query_heads,
                latent.query_rank.unwrap_or(1),
                latent.key_value_rank,
                latent.nope_head_dim,
                latent.value_head_dim,
            ];
            if dimensions.contains(&0)
                || latent.rope_head_dim == 0
                || !latent.rope_head_dim.is_multiple_of(2)
                || !latent.scale.is_finite()
                || latent.scale <= 0.0
                || latent.query_temperature.is_some_and(|temperature| !temperature.is_valid())
            {
                return Err(invalid("invalid latent attention specification"));
            }
            norm(latent.latent_normalization.epsilon)?;
            rotary(&latent.position)?;
        },
    }
    Ok(())
}

fn rotary(position: &PositionEncodingSpec) -> Result<()> {
    if let PositionEncodingSpec::Rotary(rotary) = position
        && (!rotary.theta.is_finite()
            || rotary.theta <= 0.0
            || !rotary.partial_factor.is_finite()
            || rotary.partial_factor <= 0.0)
    {
        return Err(invalid("invalid rotary position specification"));
    }
    Ok(())
}

fn state_space(state: &super::StateSpaceSpec) -> Result<()> {
    let dimensions = [
        state.heads,
        state.head_dim,
        state.state_size,
        state.groups,
        state.convolution_kernel_size,
        state.chunk_size,
    ];
    if dimensions.contains(&0) || !state.heads.is_multiple_of(state.groups) {
        return Err(invalid("invalid state-space mixer specification"));
    }
    let minimum = state.time_step_minimum;
    if !minimum.is_finite()
        || minimum < 0.0
        || state
            .time_step_maximum
            .is_some_and(|maximum| !maximum.is_finite() || maximum < minimum)
    {
        return Err(invalid("invalid state-space time step limit"));
    }
    Ok(())
}

fn feed_forward(feed_forward: &FeedForwardSpec) -> Result<()> {
    match feed_forward {
        FeedForwardSpec::Dense { intermediate_size, activation } => {
            nonzero(*intermediate_size)?;
            activation_spec(activation)
        },
        FeedForwardSpec::Routed { routed, shared } => {
            routed.validate()?;
            if let Some(shared) = shared {
                nonzero(shared.intermediate_size)?;
                activation_spec(&shared.activation)?;
            }
            Ok(())
        },
        FeedForwardSpec::DenseAndRouted {
            dense_intermediate_size,
            dense_activation,
            routed,
        } => {
            nonzero(*dense_intermediate_size)?;
            activation_spec(dense_activation)?;
            routed.validate()
        },
        FeedForwardSpec::LatentRouted { latent_size, routed, shared } => {
            if let Some(latent_size) = latent_size {
                nonzero(*latent_size)?;
            }
            routed.validate()?;
            if let Some(shared) = shared {
                nonzero(shared.intermediate_size)?;
                activation_spec(&shared.activation)?;
            }
            Ok(())
        },
        FeedForwardSpec::None => Ok(()),
    }
}

/// A clamped `SwiGLU`'s slope and limit are finite and positive.
pub(super) fn activation_spec(activation: &ActivationSpec) -> Result<()> {
    if let ActivationSpec::ClampedSwiGlu { alpha, limit } = *activation
        && !([alpha, limit].iter().all(|value| value.is_finite() && *value > 0.0))
    {
        return Err(invalid("clamped SwiGLU needs a finite positive slope and limit"));
    }
    Ok(())
}

pub(super) fn nonzero(value: usize) -> Result<()> {
    if value == 0 {
        Err(invalid("feed-forward dimensions must be non-zero"))
    } else {
        Ok(())
    }
}

pub(super) fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
