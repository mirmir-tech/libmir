use serde_json::Value;

use crate::error::{ModelsError, Result};

pub(super) fn has_fields(value: &Value, fields: &[&str]) -> bool {
    fields.iter().all(|field| value.get(*field).is_some())
}

pub(super) fn object<'a>(value: &'a Value, field: &str) -> Result<&'a Value> {
    value
        .get(field)
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid(format!("missing vision object {field}")))
}

pub(super) fn usize_field(value: &Value, field: &str) -> Result<usize> {
    let raw = value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("missing vision integer {field}")))?;
    Ok(usize::try_from(raw)?)
}

pub(super) fn optional_usize_field(value: &Value, field: &str) -> Result<Option<usize>> {
    value.get(field).map_or(Ok(None), |value| {
        value
            .as_u64()
            .ok_or_else(|| invalid(format!("invalid vision integer {field}")))
            .and_then(|raw| usize::try_from(raw).map(Some).map_err(ModelsError::from))
    })
}

pub(super) fn scalar_usize_field(value: &Value, field: &str) -> Result<usize> {
    if let Some(raw) = value.get(field).and_then(Value::as_u64) {
        return Ok(usize::try_from(raw)?);
    }
    let values = usize_array_field(value, field)?;
    values
        .first()
        .copied()
        .filter(|first| values.iter().all(|value| value == first))
        .ok_or_else(|| invalid(format!("vision field {field} must have one uniform value")))
}

pub(super) fn u32_field(value: &Value, field: &str) -> Result<u32> {
    let raw = value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("missing vision token {field}")))?;
    Ok(u32::try_from(raw)?)
}

pub(super) fn float_field(value: &Value, field: &str) -> Result<f64> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("missing vision float {field}")))
}

pub(super) fn string_field(value: &Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("missing vision string {field}")))
}

pub(super) fn bool_field(value: &Value, field: &str, default: bool) -> Result<bool> {
    value.get(field).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| invalid(format!("invalid vision boolean {field}")))
    })
}

pub(super) fn usize_array_field(value: &Value, field: &str) -> Result<Vec<usize>> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("missing vision integer array {field}")))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| invalid(format!("invalid vision integer array {field}")))
                .and_then(|value| usize::try_from(value).map_err(ModelsError::from))
        })
        .collect()
}

pub(super) fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
