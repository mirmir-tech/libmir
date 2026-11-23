mod activation;
mod experts;
mod latent;
mod rotary;
mod state_space;
mod tensors;

use activation::activation;
pub(super) use rotary::rotary;

use super::{
    AttentionOutputSpec, AttentionSpec, AttentionWindow, DecoderLayerSpec, DecoderSpec,
    FeedForwardSpec, KeyValueRelation, LinearAttentionSpec, MixerSpec, NormalizationKind,
    NormalizationSpec, PositionEncodingSpec, QkNormalization, RoutedExpertsSpec,
    RouterNormalization, SemanticModelSpec, SharedExpertSpec, model::CURRENT_SCHEMA_VERSION,
};
use crate::{
    error::{ModelsError, Result},
    layout::{AttentionLayerType, AttentionOutput, DecoderConfig, ExpertRouter},
    weights::TensorCatalog,
};

pub(super) fn compile(
    decoder: &DecoderConfig,
    tensors: &TensorCatalog,
) -> Result<SemanticModelSpec> {
    let norm = NormalizationSpec {
        kind: match decoder.norm_kind {
            crate::layout::NormKind::RmsNorm => NormalizationKind::Rms,
            crate::layout::NormKind::LayerNorm => NormalizationKind::Layer,
        },
        epsilon: decoder.rms_norm_eps,
        weight_offset: 0.0,
    };
    if let Some(hybrid) = &decoder.state_space {
        return state_space::compile(decoder, hybrid, norm, tensors);
    }
    if let Some(latent) = &decoder.latent {
        return latent::compile(decoder, latent, norm);
    }
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
            embedding_multiplier: decoder.embedding_multiplier.unwrap_or(1.0),
            residual_multiplier: decoder.residual_multiplier.unwrap_or(1.0),
            logit_divisor: decoder.logits_divisor.unwrap_or(1.0),
            layers,
            block_layout: super::BlockLayoutSpec::Unified,
            prediction_layers: prediction_layers(decoder, tensors, norm)?,
        },
    })
}

/// Prediction modules whose tensors the checkpoint carries (Qwen3.5's
/// `mtp.*`): each a decoder layer with the full-attention layers' mixer and
/// the decoder's feed-forward. A routed module without its experts, as some
/// quantizers leave it, cannot run, so the decoder loads without modules.
fn prediction_layers(
    decoder: &DecoderConfig,
    tensors: &TensorCatalog,
    norm: NormalizationSpec,
) -> Result<Vec<DecoderLayerSpec>> {
    let Some(depth) = decoder.prediction_depth else {
        return Ok(Vec::new());
    };
    let carries =
        |prefix: &str| tensors.tensors.iter().any(|tensor| tensor.name.starts_with(prefix));
    let Some(full) = (0..decoder.num_hidden_layers)
        .find(|&index| decoder.layer_type(index) == AttentionLayerType::Full)
        .filter(|_| carries("mtp."))
    else {
        return Ok(Vec::new());
    };
    let full = layer(decoder, tensors, full, norm)?;
    if full.feed_forward.routed().is_some() && !carries("mtp.layers.0.mlp.experts.") {
        return Ok(Vec::new());
    }
    Ok((0..depth.get())
        .map(|module| DecoderLayerSpec { index: module, ..full.clone() })
        .collect())
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
        norm_placement: tensors::norm_placement(decoder, tensors, index),
        mixer: mixer(decoder, tensors, index)?,
        feed_forward: if decoder.routed_experts.is_some() {
            experts::feed_forward(decoder, index)?
        } else {
            feed_forward(decoder, tensors, index)?
        },
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
    let pattern = &decoder.attention_pattern;
    let rotates = pattern.rotates(index);
    Ok(MixerSpec::SoftmaxAttention(AttentionSpec {
        query_heads: decoder.num_attention_heads,
        key_value_heads: decoder.layer_key_value_heads(index),
        head_dim,
        key_value_relation: if decoder.layer_type(index) == AttentionLayerType::Full
            && decoder.attention_k_eq_v
        {
            KeyValueRelation::KeyEqualsValue
        } else {
            KeyValueRelation::Separate
        },
        qk_normalization: if pattern.rotary_unit_qk_norm && rotates {
            QkNormalization::QueryKeyUnitRms
        } else {
            decoder.qk_normalization.unwrap_or_else(|| {
                tensors::qk_normalization(tensors, index, decoder.num_attention_heads * head_dim)
            })
        },
        projection_bias: decoder.attention_bias,
        output: attention_output(decoder.attention_output),
        sinks: decoder.attention_sinks || tensors::has_attention_sinks(tensors, index),
        scale: decoder.attention_scale.unwrap_or_else(|| head_dim_f64.sqrt().recip()),
        window: pattern
            .rotary_chunk
            .filter(|_| rotates)
            .map(|chunk| AttentionWindow::Chunked { chunk })
            .or_else(|| decoder.layer_sliding_window(index).map(AttentionWindow::Sliding)),
        position: if rotates {
            rotary(decoder, index)
        } else {
            PositionEncodingSpec::None
        },
        logit_softcap: decoder.attn_logit_softcapping,
        query_temperature: pattern.rotation_free_temperature.filter(|_| !rotates),
    }))
}

fn feed_forward(
    decoder: &DecoderConfig,
    tensors: &TensorCatalog,
    index: usize,
) -> Result<FeedForwardSpec> {
    let activation = activation(decoder);
    let routed_layer = decoder
        .routed_layer_step
        .is_none_or(|step| (index + 1).is_multiple_of(step.get()));
    let Some(expert_count) = decoder.num_experts.filter(|_| routed_layer) else {
        return Ok(FeedForwardSpec::Dense {
            intermediate_size: decoder
                .dense_intermediate_size
                .filter(|_| decoder.num_experts.is_some())
                .unwrap_or(decoder.intermediate_size),
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
        router_normalization: match decoder.expert_router {
            ExpertRouter::SoftmaxTopK => RouterNormalization::SoftmaxTopK,
            ExpertRouter::SparseMixer { jitter } => {
                RouterNormalization::SparseMixer { jitter_eps: jitter }
            },
            ExpertRouter::SigmoidInputScaled => RouterNormalization::SigmoidInputScaledTopK,
        },
        group_limit: None,
        output_scale: 1.0,
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
                // A sigmoid gate on the shared output where the checkpoint
                // has one (Qwen2-MoE); Llama 4's adds ungated.
                gated_output: tensors::has_shared_expert_gate(tensors, index),
            });
    Ok(FeedForwardSpec::Routed { routed, shared })
}

const fn attention_output(output: AttentionOutput) -> AttentionOutputSpec {
    match output {
        AttentionOutput::Direct => AttentionOutputSpec::Direct,
        AttentionOutput::Gated => AttentionOutputSpec::Gated,
    }
}

pub(super) fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
