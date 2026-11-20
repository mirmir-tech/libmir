use std::num::NonZeroUsize;

use serde_json::Value;

use super::{
    ExpertRouting, LatentAttentionConfig, LatentDecoderConfig, LatentExpertsConfig, RotaryPairing,
};
use crate::{
    error::Result,
    layout::decoder::{
        parse::{
            invalid, optional_bool, optional_f64, optional_string, optional_usize, require_usize,
        },
        rope,
    },
};

/// Reads a latent-attention decoder, recognized by its `kv_lora_rank`.
pub(in crate::layout::decoder) fn latent_decoder(
    decoder: &Value,
) -> Result<Option<LatentDecoderConfig>> {
    let Some(key_value_rank) = optional_usize(decoder, &["kv_lora_rank"])? else {
        return Ok(None);
    };
    let attention = LatentAttentionConfig {
        query_rank: optional_usize(decoder, &["q_lora_rank"])?.map(positive).transpose()?,
        key_value_rank: positive(key_value_rank)?.get(),
        nope_head_dim: positive(require_usize(decoder, &["qk_nope_head_dim"])?)?.get(),
        rope_head_dim: even(require_usize(decoder, &["qk_rope_head_dim"])?)?,
        value_head_dim: positive(require_usize(decoder, &["v_head_dim"])?)?.get(),
        rotary_pairing: if optional_bool(decoder, "rope_interleave")?.unwrap_or(true) {
            RotaryPairing::Interleaved
        } else {
            RotaryPairing::Halves
        },
        scale: 1.0,
    };
    let query_width = f64::from(u32::try_from(attention.query_head_dim())?);
    let scale = query_width.sqrt().recip() * mscale_all_dim(decoder)?.powi(2);
    Ok(Some(LatentDecoderConfig {
        attention: LatentAttentionConfig { scale, ..attention },
        experts: experts(decoder)?,
    }))
}

/// `YaRN`'s attention-temperature correction applied to the whole softmax:
/// `0.1 · mscale_all_dim · ln(factor) + 1`, as `DeepSeek`'s reference code
/// does.
fn mscale_all_dim(decoder: &Value) -> Result<f64> {
    let Some(scaling) = rope::scaling_value(decoder) else {
        return Ok(1.0);
    };
    let factor = optional_f64(scaling, &["factor"])?.unwrap_or(1.0);
    let all_dim = optional_f64(scaling, &["mscale_all_dim"])?.unwrap_or(0.0);
    if !factor.is_finite() || !all_dim.is_finite() {
        return Err(invalid("rope_scaling factor and mscale_all_dim must be finite"));
    }
    if all_dim == 0.0 || factor <= 1.0 {
        return Ok(1.0);
    }
    Ok(0.1_f64.mul_add(all_dim * factor.ln(), 1.0))
}

fn experts(decoder: &Value) -> Result<Option<LatentExpertsConfig>> {
    if optional_usize(decoder, &["n_routed_experts", "num_experts"])?.is_none() {
        return Ok(None);
    }
    if optional_usize(decoder, &["moe_layer_freq"])?.unwrap_or(1) != 1 {
        return Err(invalid("routed layers interleaved by moe_layer_freq are not supported"));
    }
    let groups = optional_usize(decoder, &["n_group", "num_expert_group"])?.unwrap_or(1);
    let selected_groups = optional_usize(decoder, &["topk_group"])?.unwrap_or(1);
    if groups > 1 || selected_groups > 1 {
        return Err(invalid("grouped expert selection (n_group > 1) is not supported"));
    }
    let output_scale = optional_f64(decoder, &["routed_scaling_factor"])?.unwrap_or(1.0);
    if !output_scale.is_finite() || output_scale <= 0.0 {
        return Err(invalid("routed_scaling_factor must be finite and greater than zero"));
    }
    Ok(Some(LatentExpertsConfig {
        first_dense_layers: optional_usize(decoder, &["first_k_dense_replace"])?.unwrap_or(0),
        shared_experts: optional_usize(decoder, &["n_shared_experts", "num_shared_experts"])?
            .and_then(NonZeroUsize::new),
        routing: routing(decoder)?,
        output_scale,
    }))
}

fn routing(decoder: &Value) -> Result<ExpertRouting> {
    let scoring = match optional_string(decoder, "scoring_func")? {
        Some(scoring) => Some(scoring),
        None => optional_string(decoder, "moe_router_activation_func")?,
    };
    let renormalize = match optional_bool(decoder, "norm_topk_prob")? {
        Some(renormalize) => renormalize,
        None => optional_bool(decoder, "moe_renormalize")?.unwrap_or(false),
    };
    match scoring.as_deref() {
        None | Some("softmax") => Ok(ExpertRouting::Softmax { renormalize }),
        Some("sigmoid") => {
            let method = optional_string(decoder, "topk_method")?;
            if !matches!(method.as_deref(), None | Some("noaux_tc")) {
                return Err(invalid("sigmoid routing requires topk_method noaux_tc"));
            }
            if !renormalize {
                return Err(invalid("sigmoid routing without renormalization is not supported"));
            }
            Ok(ExpertRouting::SigmoidWithCorrectionBias)
        },
        Some(other) => Err(invalid(format!("unsupported expert scoring function {other}"))),
    }
}

fn positive(value: usize) -> Result<NonZeroUsize> {
    NonZeroUsize::new(value).ok_or_else(|| invalid("latent attention dimensions must be positive"))
}

fn even(value: usize) -> Result<usize> {
    if value == 0 || !value.is_multiple_of(2) {
        return Err(invalid("qk_rope_head_dim must be positive and even"));
    }
    Ok(value)
}
