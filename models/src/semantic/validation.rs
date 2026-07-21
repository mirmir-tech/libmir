use super::{
    ActivationSpec, FeedForwardSpec, MixerSpec, PositionEncodingSpec, SemanticModelSpec,
    model::CURRENT_SCHEMA_VERSION,
};
use crate::error::{ModelsError, Result};

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
    norm(decoder.final_norm.epsilon)?;
    for (expected, layer) in decoder.layers.iter().enumerate() {
        if layer.index != expected {
            return Err(invalid("decoder layer indices must be contiguous and ordered"));
        }
        norm(layer.input_norm.epsilon)?;
        norm(layer.post_attention_norm.epsilon)?;
        mixer(&layer.mixer)?;
        feed_forward(&layer.feed_forward)?;
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
                || attention.window == Some(0)
            {
                return Err(invalid("invalid softmax attention specification"));
            }
            if let PositionEncodingSpec::Rotary(rotary) = &attention.position
                && (!rotary.theta.is_finite()
                    || rotary.theta <= 0.0
                    || !rotary.partial_factor.is_finite()
                    || rotary.partial_factor <= 0.0)
            {
                return Err(invalid("invalid rotary position specification"));
            }
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
            routed_spec(routed)?;
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
            routed_spec(routed)
        },
    }
}

fn routed_spec(routed: &super::RoutedExpertsSpec) -> Result<()> {
    nonzero(routed.expert_count)?;
    nonzero(routed.top_k)?;
    nonzero(routed.intermediate_size)?;
    if routed.top_k > routed.expert_count {
        return Err(invalid("routed top-k exceeds expert count"));
    }
    activation_spec(&routed.activation)
}

fn activation_spec(activation: &ActivationSpec) -> Result<()> {
    if let ActivationSpec::SwiGlu { alpha, clamp, up_shift } = activation
        && (!alpha.is_finite()
            || *alpha <= 0.0
            || clamp.is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
            || !up_shift.is_finite())
    {
        return Err(invalid("invalid SwiGLU activation specification"));
    }
    Ok(())
}

fn nonzero(value: usize) -> Result<()> {
    if value == 0 {
        Err(invalid("feed-forward dimensions must be non-zero"))
    } else {
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
