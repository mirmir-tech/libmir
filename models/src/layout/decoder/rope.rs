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
}

impl RopeScaling {
    #[must_use]
    pub const fn piecewise_frequency(self) -> (f64, f64, f64, usize) {
        match self {
            Self::PiecewiseFrequency {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_context_len,
            } => (factor, low_frequency_factor, high_frequency_factor, original_context_len),
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
