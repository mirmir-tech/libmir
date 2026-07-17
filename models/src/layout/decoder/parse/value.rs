use serde_json::Value;

use crate::error::{ModelsError, Result};

pub(in crate::layout::decoder) fn require_usize(value: &Value, keys: &[&str]) -> Result<usize> {
    optional_usize(value, keys)?.ok_or_else(|| invalid(format!("missing {}", keys.join(" or "))))
}

pub(in crate::layout::decoder) fn optional_usize(
    value: &Value,
    keys: &[&str],
) -> Result<Option<usize>> {
    for key in keys {
        let Some(raw) = value.get(*key) else {
            continue;
        };
        if raw.is_null() {
            return Ok(None);
        }
        let Some(number) = raw.as_u64() else {
            return Err(invalid(format!("{key} must be an unsigned integer")));
        };
        return Ok(Some(usize::try_from(number)?));
    }
    Ok(None)
}

pub(super) fn optional_f64(value: &Value, keys: &[&str]) -> Result<Option<f64>> {
    for key in keys {
        let Some(raw) = value.get(*key) else {
            continue;
        };
        if raw.is_null() {
            return Ok(None);
        }
        let Some(number) = raw.as_f64() else {
            return Err(invalid(format!("{key} must be a number")));
        };
        return Ok(Some(number));
    }
    Ok(None)
}

pub(super) fn optional_bool(value: &Value, key: &str) -> Result<Option<bool>> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    raw.as_bool()
        .map(Some)
        .ok_or_else(|| invalid(format!("{key} must be a boolean")))
}

pub(super) fn optional_string(value: &Value, key: &str) -> Result<Option<String>> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    raw.as_str()
        .map(|text| Some(text.to_owned()))
        .ok_or_else(|| invalid(format!("{key} must be a string")))
}

pub(super) fn rope_theta(value: &Value) -> Result<Option<f64>> {
    if let Some(theta) = optional_f64(value, &["rope_theta", "rotary_emb_base"])? {
        return Ok(Some(theta));
    }
    let Some(parameters) = value.get("rope_parameters") else {
        return Ok(None);
    };
    if let Some(theta) = parameters.get("rope_theta").and_then(Value::as_f64) {
        return Ok(Some(theta));
    }
    Ok(parameters
        .as_object()
        .into_iter()
        .flat_map(serde_json::Map::values)
        .find_map(|item| item.get("rope_theta").and_then(Value::as_f64)))
}

pub(super) fn partial_rotary_factor(value: &Value) -> Result<Option<f64>> {
    if let Some(factor) = optional_f64(value, &["partial_rotary_factor"])? {
        return Ok(Some(factor));
    }
    let Some(parameters) = value.get("rope_parameters") else {
        return Ok(None);
    };
    let Some(raw) = parameters.get("partial_rotary_factor") else {
        return Ok(None);
    };
    raw.as_f64()
        .map(Some)
        .ok_or_else(|| invalid("rope partial_rotary_factor must be a number"))
}

pub(in crate::layout::decoder) fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
