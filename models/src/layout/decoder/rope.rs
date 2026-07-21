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
    pub const fn yarn(self) -> Option<(f64, f64, f64, usize)> {
        match self {
            Self::Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_context_len,
            } => Some((factor, beta_fast, beta_slow, original_context_len)),
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
    let beta_fast = number(value, "beta_fast")?;
    let beta_slow = number(value, "beta_slow")?;
    let original_context_len = integer(value, "original_max_position_embeddings")?;
    if !factor.is_finite()
        || !beta_fast.is_finite()
        || !beta_slow.is_finite()
        || factor <= 0.0
        || beta_fast <= beta_slow
        || beta_slow <= 0.0
        || original_context_len == 0
    {
        return Err(invalid("invalid YaRN rope_scaling parameters"));
    }
    Ok(RopeScaling::Yarn {
        factor,
        beta_fast,
        beta_slow,
        original_context_len,
    })
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

fn integer(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .map(|value| -> Result<usize> { Ok(usize::try_from(value)?) })
        .transpose()?
        .ok_or_else(|| invalid(format!("rope_scaling.{field} must be an unsigned integer")))
}
