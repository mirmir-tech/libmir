use std::num::NonZeroUsize;

use serde_json::Value;

use super::{LatentAttentionConfig, LatentDecoderConfig, QueryTemperature, RotaryPairing};
use crate::{
    error::Result,
    layout::decoder::{
        parse::{invalid, optional_bool, optional_f64, optional_usize, require_usize},
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
        query_temperature: query_temperature(decoder)?,
    };
    let query_width = f64::from(u32::try_from(attention.query_head_dim())?);
    let scale = query_width.sqrt().recip() * mscale_all_dim(decoder)?.powi(2);
    Ok(Some(LatentDecoderConfig {
        attention: LatentAttentionConfig { scale, ..attention },
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

/// Mistral 4's `llama_4_scaling_beta`, which scales queries past every
/// `original_max_position_embeddings` window; zero or absent leaves them.
fn query_temperature(decoder: &Value) -> Result<Option<QueryTemperature>> {
    let Some(scaling) = rope::scaling_value(decoder) else {
        return Ok(None);
    };
    let Some(beta) = optional_f64(scaling, &["llama_4_scaling_beta"])?.filter(|beta| *beta != 0.0)
    else {
        return Ok(None);
    };
    let window = optional_usize(scaling, &["original_max_position_embeddings"])?
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| invalid("llama_4_scaling_beta requires original_max_position_embeddings"))?;
    if !beta.is_finite() {
        return Err(invalid("llama_4_scaling_beta must be finite"));
    }
    Ok(Some(QueryTemperature { beta, window, offset: 0 }))
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
