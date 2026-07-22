use super::{
    ActivationSpec, AttentionOutputSpec, AttentionSpec, DecoderLayerSpec, DecoderSpec,
    FeedForwardSpec, KeyValueRelation, LinearAttentionSpec, MixerSpec, NormalizationKind,
    NormalizationSpec, PositionEncodingSpec, QkNormalization, RopeScalingSpec, RotaryLayoutSpec,
    RotarySpec, RoutedExpertsSpec, RouterNormalization, SemanticModelSpec, SharedExpertSpec,
    model::CURRENT_SCHEMA_VERSION,
};
use crate::{
    error::{ModelsError, Result},
    layout::{
        AttentionLayerType, AttentionOutput, DecoderConfig, RopeScaling, RotaryEmbeddingLayout,
    },
    weights::TensorCatalog,
};

pub(super) fn compile(
    decoder: &DecoderConfig,
    tensors: &TensorCatalog,
) -> Result<SemanticModelSpec> {
    let norm = NormalizationSpec {
        kind: NormalizationKind::Rms,
        epsilon: decoder.rms_norm_eps,
    };
    let layers = (0..decoder.num_hidden_layers)
        .map(|index| layer(decoder, tensors, index, norm))
        .collect::<Result<Vec<_>>>()?;
    Ok(SemanticModelSpec {
        schema_version: CURRENT_SCHEMA_VERSION,
        decoder: DecoderSpec {
            hidden_size: decoder.hidden_size,
            vocab_size: decoder.vocab_size,
            tie_word_embeddings: decoder.tie_word_embeddings,
            final_norm: norm,
            layers,
        },
    })
}

fn layer(
    decoder: &DecoderConfig,
    tensors: &TensorCatalog,
    index: usize,
    norm: NormalizationSpec,
) -> Result<DecoderLayerSpec> {
    Ok(DecoderLayerSpec {
        index,
        input_norm: norm,
        post_attention_norm: norm,
        mixer: mixer(decoder, tensors, index)?,
        feed_forward: feed_forward(decoder)?,
    })
}

fn mixer(decoder: &DecoderConfig, tensors: &TensorCatalog, index: usize) -> Result<MixerSpec> {
    if decoder.layer_type(index) == AttentionLayerType::Linear {
        let config = decoder
            .linear_attention
            .as_ref()
            .ok_or_else(|| invalid("linear layer is missing linear attention configuration"))?;
        return Ok(MixerSpec::LinearAttention(LinearAttentionSpec {
            convolution_kernel_size: config.convolution_kernel_size,
            key_heads: config.key_heads,
            value_heads: config.value_heads,
            key_head_dim: config.key_head_dim,
            value_head_dim: config.value_head_dim,
            output: attention_output(decoder.attention_output),
        }));
    }
    let head_dim = decoder.layer_head_dim(index);
    let head_dim_f64 = f64::from(u32::try_from(head_dim)?);
    Ok(MixerSpec::SoftmaxAttention(AttentionSpec {
        query_heads: decoder.num_attention_heads,
        key_value_heads: decoder.layer_key_value_heads(index),
        head_dim,
        key_value_relation: if decoder.attention_k_eq_v {
            KeyValueRelation::KeyEqualsValue
        } else {
            KeyValueRelation::Separate
        },
        qk_normalization: if has_qk_norm(tensors, index) {
            QkNormalization::QueryKeyRms
        } else {
            QkNormalization::None
        },
        projection_bias: decoder.attention_bias,
        output: attention_output(decoder.attention_output),
        sinks: decoder.attention_sinks || has_attention_sinks(tensors, index),
        scale: decoder.attention_scale.unwrap_or_else(|| head_dim_f64.sqrt().recip()),
        window: decoder.layer_sliding_window(index),
        position: rotary(decoder, index),
    }))
}

fn rotary(decoder: &DecoderConfig, index: usize) -> PositionEncodingSpec {
    PositionEncodingSpec::Rotary(RotarySpec {
        theta: decoder.rope_theta_for_layer(index).unwrap_or(10_000.0),
        partial_factor: decoder.partial_rotary_factor_for_layer(index).unwrap_or(1.0),
        layout: match &decoder.rope_layout {
            RotaryEmbeddingLayout::Standard => RotaryLayoutSpec::Standard,
            RotaryEmbeddingLayout::MultiSection(sections) => {
                RotaryLayoutSpec::MultiSection(sections.clone())
            },
            RotaryEmbeddingLayout::InterleavedMultiSection(sections) => {
                RotaryLayoutSpec::InterleavedMultiSection(sections.clone())
            },
        },
        algorithm: decoder.rope_type_for_layer(index).map(str::to_owned),
        scaling: decoder.rope_scaling.map(rope_scaling),
    })
}

fn rope_scaling(scaling: RopeScaling) -> RopeScalingSpec {
    match scaling {
        RopeScaling::PiecewiseFrequency {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_context_len,
        } => RopeScalingSpec::PiecewiseFrequency {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_context_len,
        },
        RopeScaling::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_context_len,
            attention_factor,
        } => RopeScalingSpec::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_context_len,
            attention_factor,
        },
    }
}

fn feed_forward(decoder: &DecoderConfig) -> Result<FeedForwardSpec> {
    let activation = activation(decoder);
    let Some(expert_count) = decoder.num_experts else {
        return Ok(FeedForwardSpec::Dense {
            intermediate_size: decoder.intermediate_size,
            activation,
        });
    };
    let top_k = decoder
        .top_k_experts
        .ok_or_else(|| invalid("routed feed-forward is missing top-k expert count"))?;
    let routed = RoutedExpertsSpec {
        expert_count,
        top_k,
        intermediate_size: decoder.moe_intermediate_size.unwrap_or(decoder.intermediate_size),
        activation: activation.clone(),
        router_normalization: RouterNormalization::SoftmaxTopK,
    };
    if decoder.attention_k_eq_v && decoder.hidden_activation.as_deref() == Some("gelu_pytorch_tanh")
    {
        return Ok(FeedForwardSpec::DenseAndRouted {
            dense_intermediate_size: decoder.intermediate_size,
            dense_activation: activation,
            routed,
        });
    }
    let shared =
        decoder
            .shared_expert_intermediate_size
            .map(|intermediate_size| SharedExpertSpec {
                intermediate_size,
                activation,
                gated_output: true,
            });
    Ok(FeedForwardSpec::Routed { routed, shared })
}

fn activation(decoder: &DecoderConfig) -> ActivationSpec {
    match decoder.hidden_activation.as_deref() {
        Some("gelu_pytorch_tanh") => ActivationSpec::GeluTanh,
        Some("silu") | None => ActivationSpec::SwiGlu {
            alpha: if decoder.swiglu_limit.is_some() {
                1.702
            } else {
                1.0
            },
            clamp: decoder.swiglu_limit,
            up_shift: if decoder.swiglu_limit.is_some() {
                1.0
            } else {
                0.0
            },
        },
        Some(name) => ActivationSpec::NamedGated { name: name.to_owned() },
    }
}

const fn attention_output(output: AttentionOutput) -> AttentionOutputSpec {
    match output {
        AttentionOutput::Direct => AttentionOutputSpec::Direct,
        AttentionOutput::Gated => AttentionOutputSpec::Gated,
    }
}

fn has_qk_norm(tensors: &TensorCatalog, index: usize) -> bool {
    layer_roots(index).into_iter().any(|root| {
        tensors.contains(&format!("{root}.self_attn.q_norm.weight"))
            && tensors.contains(&format!("{root}.self_attn.k_norm.weight"))
    })
}

fn has_attention_sinks(tensors: &TensorCatalog, index: usize) -> bool {
    layer_roots(index)
        .into_iter()
        .any(|root| tensors.contains(&format!("{root}.self_attn.sinks")))
}

fn layer_roots(index: usize) -> [String; 4] {
    [
        format!("model.layers.{index}"),
        format!("layers.{index}"),
        format!("language_model.model.layers.{index}"),
        format!("model.language_model.layers.{index}"),
    ]
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
