use serde_json::Value;

use super::parse::invalid;
use crate::error::Result;

mod long;

pub use long::RopeFactors;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// Every frequency divided by `factor` (`rope_type = linear`).
    Linear { factor: f64 },
    PiecewiseFrequency {
        factor: f64,
        low_frequency_factor: f64,
        high_frequency_factor: f64,
        original_context_len: usize,
    },
    Yarn {
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        original_context_len: usize,
        attention_factor: f64,
        truncate: bool,
    },
    /// Every pair's period multiplied by its factor and cosine and sine by
    /// `attention_factor` (Phi-3 `LongRoPE`).
    LongRope {
        factors: RopeFactors,
        original_context_len: usize,
        attention_factor: f64,
    },
}

impl RopeScaling {
    /// The factor on cosine and sine: `YaRN`'s and `LongRoPE`'s, else 1.
    #[must_use]
    pub const fn attention_factor(self) -> f64 {
        match self {
            Self::Yarn { attention_factor, .. } | Self::LongRope { attention_factor, .. } => {
                attention_factor
            },
            Self::Linear { .. } | Self::PiecewiseFrequency { .. } => 1.0,
        }
    }

    #[must_use]
    pub const fn piecewise_frequency(self) -> Option<(f64, f64, f64, usize)> {
        match self {
            Self::PiecewiseFrequency {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_context_len,
            } => Some((factor, low_frequency_factor, high_frequency_factor, original_context_len)),
            Self::Linear { .. } | Self::Yarn { .. } | Self::LongRope { .. } => None,
        }
    }

    #[must_use]
    pub const fn yarn(self) -> Option<(f64, f64, f64, usize, f64, bool)> {
        match self {
            Self::Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_context_len,
                attention_factor,
                truncate,
            } => Some((
                factor,
                beta_fast,
                beta_slow,
                original_context_len,
                attention_factor,
                truncate,
            )),
            Self::Linear { .. } | Self::PiecewiseFrequency { .. } | Self::LongRope { .. } => None,
        }
    }
}

/// The rotary scaling description: `rope_scaling`, or a flat
/// `rope_parameters` (Transformers 5) naming a scaled rope type. Per-layer-type
/// `rope_parameters` (Gemma) and unscaled types stay out of scope here.
pub(super) fn scaling_value(config: &Value) -> Option<&Value> {
    if let Some(value) = config.get("rope_scaling").filter(|value| !value.is_null()) {
        return Some(value);
    }
    config.get("rope_parameters").filter(|parameters| {
        rope_type_of(parameters)
            .is_some_and(|kind| matches!(kind, "yarn" | "linear" | "llama3" | "longrope"))
    })
}

/// `rope_type`, or the older `type` key `DeepSeek` V2 checkpoints write.
fn rope_type_of(value: &Value) -> Option<&str> {
    value.get("rope_type").or_else(|| value.get("type")).and_then(Value::as_str)
}

pub(super) fn scaling(config: &Value) -> Result<Option<RopeScaling>> {
    let Some(value) = scaling_value(config) else {
        return Ok(None);
    };
    let rope_type =
        rope_type_of(value).ok_or_else(|| invalid("rope_scaling.rope_type must be a string"))?;
    if rope_type == "yarn" {
        return yarn(value).map(Some);
    }
    if matches!(rope_type, "longrope" | "su") {
        return long::long_rope(config, value).map(Some);
    }
    if rope_type == "linear" {
        let factor = number(value, "factor")?;
        if !factor.is_finite() || factor <= 0.0 {
            return Err(invalid("invalid linear rope_scaling factor"));
        }
        return Ok(Some(RopeScaling::Linear { factor }));
    }
    if rope_type != "llama3" {
        return Err(invalid(format!("unsupported rope_scaling type {rope_type}")));
    }
    let factor = number(value, "factor")?;
    let low_frequency_factor = number(value, "low_freq_factor")?;
    let high_frequency_factor = number(value, "high_freq_factor")?;
    let original_context_len = integer(value, "original_max_position_embeddings")?;
    if !factor.is_finite()
        || !low_frequency_factor.is_finite()
        || !high_frequency_factor.is_finite()
        || factor <= 0.0
        || low_frequency_factor <= 0.0
        || high_frequency_factor < low_frequency_factor
        || original_context_len == 0
    {
        return Err(invalid("invalid piecewise rope_scaling parameters"));
    }
    Ok(Some(RopeScaling::PiecewiseFrequency {
        factor,
        low_frequency_factor,
        high_frequency_factor,
        original_context_len,
    }))
}

fn yarn(value: &Value) -> Result<RopeScaling> {
    let factor = number(value, "factor")?;
    let beta_fast = optional_number(value, "beta_fast")?.unwrap_or(32.0);
    let beta_slow = optional_number(value, "beta_slow")?.unwrap_or(1.0);
    let original_context_len = integer(value, "original_max_position_embeddings")?;
    let attention_factor = yarn_attention_factor(value, factor)?;
    let truncate = value.get("truncate").map_or(Ok(true), |value| {
        value
            .as_bool()
            .ok_or_else(|| invalid("rope_scaling.truncate must be a boolean"))
    })?;
    if !factor.is_finite()
        || !beta_fast.is_finite()
        || !beta_slow.is_finite()
        || factor <= 0.0
        || beta_fast <= beta_slow
        || beta_slow <= 0.0
        || original_context_len == 0
        || !attention_factor.is_finite()
        || attention_factor <= 0.0
    {
        return Err(invalid("invalid YaRN rope_scaling parameters"));
    }
    Ok(RopeScaling::Yarn {
        factor,
        beta_fast,
        beta_slow,
        original_context_len,
        attention_factor,
        truncate,
    })
}

fn yarn_attention_factor(value: &Value, factor: f64) -> Result<f64> {
    if let Some(attention_factor) = optional_number(value, "attention_factor")? {
        return Ok(attention_factor);
    }
    let default = yarn_mscale(factor, 1.0);
    if let Some(multiplier) = optional_number(value, "attn_factor")? {
        return Ok(default * multiplier);
    }
    match (optional_number(value, "mscale")?, optional_number(value, "mscale_all_dim")?) {
        (Some(mscale), Some(all_dim)) => {
            Ok(yarn_mscale(factor, mscale) / yarn_mscale(factor, all_dim))
        },
        _ => Ok(default),
    }
}

fn yarn_mscale(factor: f64, scale: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        0.1_f64.mul_add(scale * factor.ln(), 1.0)
    }
}

fn number(value: &Value, field: &str) -> Result<f64> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("rope_scaling.{field} must be a number")))
}

pub(super) fn optional_number(value: &Value, field: &str) -> Result<Option<f64>> {
    value
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| invalid(format!("rope_scaling.{field} must be a number")))
        })
        .transpose()
}

pub(super) fn integer(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .map(|value| -> Result<usize> { Ok(usize::try_from(value)?) })
        .transpose()?
        .ok_or_else(|| invalid(format!("rope_scaling.{field} must be an unsigned integer")))
}
