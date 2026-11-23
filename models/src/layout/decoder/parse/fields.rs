use serde_json::Value;

use super::{
    QUERY_KEY_ROWS, QUERY_KEY_ROWS_INTERLEAVED, invalid, optional_f64, optional_string,
    optional_usize,
};
use crate::{
    error::Result,
    layout::decoder::{AttentionLayerType, StateSpaceHybridConfig},
};

pub(super) fn intermediate_size(value: &Value) -> Result<usize> {
    optional_usize(
        value,
        &[
            "intermediate_size",
            "ffn_hidden_size",
            "n_inner",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ],
    )?
    .ok_or_else(|| {
        invalid("missing intermediate_size or ffn_hidden_size or n_inner or MoE intermediate size")
    })
}

pub(super) fn hybrid_layer_types(hybrid: &StateSpaceHybridConfig) -> Vec<AttentionLayerType> {
    hybrid
        .blocks
        .layers()
        .iter()
        .map(|layer| {
            if layer.is_state_space() {
                AttentionLayerType::StateSpace
            } else {
                AttentionLayerType::Full
            }
        })
        .collect()
}

pub(super) fn resolve_head_dim(
    decoder: &Value,
    hidden_size: usize,
    num_attention_heads: usize,
) -> Result<usize> {
    if let Some(head_dim) =
        optional_usize(decoder, &["head_dim", "attention_head_dim", "kv_channels"])?
    {
        return Ok(head_dim);
    }
    if !hidden_size.is_multiple_of(num_attention_heads) {
        return Err(invalid("hidden_size must be divisible by num_attention_heads"));
    }
    Ok(hidden_size / num_attention_heads)
}

pub(super) fn decoder_value(value: &Value) -> &Value {
    ["text_config", "language_config", "llm_config"]
        .into_iter()
        .filter_map(|key| value.get(key))
        .find(|section| section.is_object())
        .unwrap_or(value)
}

pub(super) fn attention_scale(decoder: &Value) -> Result<Option<f64>> {
    // Gemma: scores scale by `query_pre_attn_scalar ** -0.5`.
    Ok(
        optional_f64(decoder, &["attention_scale", "attention_multiplier"])?.or(optional_f64(
            decoder,
            &["query_pre_attn_scalar"],
        )?
        .map(|scalar| scalar.powf(-0.5))),
    )
}

pub(super) fn query_key_rows_interleaved(decoder: &Value) -> Result<bool> {
    match optional_string(decoder, QUERY_KEY_ROWS)?.as_deref() {
        None => Ok(false),
        Some(QUERY_KEY_ROWS_INTERLEAVED) => Ok(true),
        Some(other) => Err(invalid(format!("unknown {QUERY_KEY_ROWS} `{other}`"))),
    }
}

pub(super) fn max_position_embeddings(decoder: &Value) -> Result<usize> {
    let names = ["max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"];
    Ok(optional_usize(decoder, &names)?.unwrap_or(4096))
}

pub(super) fn rms_norm_eps(decoder: &Value) -> Result<f64> {
    let names = [
        "rms_norm_eps",
        "layer_norm_epsilon",
        "layer_norm_eps",
        "layernorm_epsilon",
        "norm_epsilon",
    ];
    Ok(optional_f64(decoder, &names)?.unwrap_or(1e-5))
}

/// Granite divides logits by `logits_scaling`; Cohere multiplies them by
/// `logit_scale`, which divides by its reciprocal.
pub(super) fn logits_divisor(decoder: &Value) -> Result<Option<f64>> {
    Ok(match optional_f64(decoder, &["logits_scaling"])? {
        Some(divisor) => Some(divisor),
        None => optional_f64(decoder, &["logit_scale"])?
            .filter(|scale| *scale > 0.0)
            .map(f64::recip),
    })
}

/// Tied embeddings, declared by the text decoder or its multimodal parent.
pub(super) fn tie_word_embeddings(value: &Value, decoder: &Value) -> Result<bool> {
    Ok(super::optional_bool(decoder, "tie_word_embeddings")?
        .or(super::optional_bool(value, "tie_word_embeddings")?)
        .unwrap_or(false))
}

/// Llama 4's `interleave_moe_layer_step`: routed experts on every `step`-th
/// layer.
pub(super) fn routed_layer_step(decoder: &Value) -> Result<Option<std::num::NonZeroUsize>> {
    optional_usize(decoder, &["interleave_moe_layer_step"])?
        .map(|step| {
            std::num::NonZeroUsize::new(step)
                .ok_or_else(|| invalid("interleave_moe_layer_step must be positive"))
        })
        .transpose()
}

/// The shared expert's width: declared, or Llama 4's one shared expert at
/// the routed experts' `intermediate_size`.
pub(super) fn shared_expert_intermediate_size(
    decoder: &Value,
    defaults: &super::super::defaults::Defaults,
) -> Result<Option<usize>> {
    if let Some(width) = optional_usize(decoder, &["shared_expert_intermediate_size"])? {
        return Ok(Some(width));
    }
    Ok(if defaults.router == super::super::defaults::Router::SigmoidInputScaled {
        Some(super::super::parse::require_usize(decoder, &["intermediate_size"])?)
    } else {
        None
    })
}

/// Prediction modules of a decoder that is not a state-space hybrid.
pub(super) fn prediction_depth(
    decoder: &Value,
    state_space: bool,
) -> Result<Option<std::num::NonZeroUsize>> {
    if state_space {
        return Ok(None);
    }
    Ok(optional_usize(decoder, &["mtp_num_hidden_layers", "num_nextn_predict_layers"])?
        .and_then(std::num::NonZeroUsize::new))
}
