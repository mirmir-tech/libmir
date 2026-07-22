use serde_json::Value;

use super::parse::invalid;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
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
    },
}

impl RopeScaling {
    #[must_use]
    pub const fn piecewise_frequency(self) -> Option<(f64, f64, f64, usize)> {
        match self {
            Self::PiecewiseFrequency {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_context_len,
            } => Some((factor, low_frequency_factor, high_frequency_factor, original_context_len)),
            Self::Yarn { .. } => None,
        }
    }

    #[must_use]
    pub const fn yarn(self) -> Option<(f64, f64, f64, usize, f64)> {
        match self {
            Self::Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_context_len,
                attention_factor,
            } => Some((factor, beta_fast, beta_slow, original_context_len, attention_factor)),
            Self::PiecewiseFrequency { .. } => None,
        }
    }
}

pub(super) fn scaling(config: &Value) -> Result<Option<RopeScaling>> {
    let Some(value) = config.get("rope_scaling") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let rope_type = string(value, "rope_type")?;
    if rope_type == "yarn" {
        return yarn(value).map(Some);
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
        || high_frequency_factor <= low_frequency_factor
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

fn string(value: &Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(Into::into)
        .ok_or_else(|| invalid(format!("rope_scaling.{field} must be a string")))
}

fn number(value: &Value, field: &str) -> Result<f64> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("rope_scaling.{field} must be a number")))
}

fn optional_number(value: &Value, field: &str) -> Result<Option<f64>> {
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

fn integer(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .map(|value| -> Result<usize> { Ok(usize::try_from(value)?) })
        .transpose()?
        .ok_or_else(|| invalid(format!("rope_scaling.{field} must be an unsigned integer")))
}
