//! The JSON Schema subset a decision schema accepts. Every other keyword is
//! rejected, so an accepted schema never yields a value it would refuse.

use serde_json::{Map, Value};

use super::invalid;
use crate::error::Result;

/// Keywords that describe but never constrain a value; `format` is an
/// annotation by default and `nullable` only widens what is allowed.
const ANNOTATIONS: [&str; 11] = [
    "title", "description", "default", "examples", "$comment", "deprecated", "$id", "readOnly",
    "writeOnly", "format", "nullable",
];
/// Top-level keywords besides annotations; `$defs` is inert without `$ref`.
const TOP_LEVEL: [&str; 6] =
    ["type", "properties", "required", "additionalProperties", "$schema", "$defs"];
/// Property keywords besides annotations.
const PROPERTY: [&str; 7] = ["const", "enum", "type", "minimum", "maximum", "anyOf", "oneOf"];

/// Checks the top level; every answer sets every property, so `required`
/// only has to name declared properties.
pub(super) fn check_top_level(
    object: &Map<String, Value>,
    properties: &Map<String, Value>,
) -> Result<()> {
    unsupported("", object, &TOP_LEVEL)?;
    let Some(required) = object.get("required") else {
        return Ok(());
    };
    let names = required.as_array().ok_or_else(|| invalid("'required' must be an array"))?;
    names
        .iter()
        .find(|name| !name.as_str().is_some_and(|name| properties.contains_key(name)))
        .map_or(Ok(()), |name| {
            Err(invalid(format!("'required' names {name}, which is not a property")))
        })
}

/// Checks one property: a union stands alone, `const` and `enum` exclude each
/// other and must match `type`, and bounds belong only to an integer scale.
pub(super) fn check_property(path: &str, object: &Map<String, Value>) -> Result<()> {
    unsupported(&format!("{path}: "), object, &PROPERTY)?;
    let has = |key: &str| object.contains_key(key);
    if (has("anyOf") || has("oneOf"))
        && let Some(key) = ["anyOf", "oneOf", "const", "enum", "type", "minimum", "maximum"]
            .into_iter()
            .filter(|key| has(key))
            .nth(1)
    {
        return Err(invalid(format!("{path}: a union cannot be combined with '{key}'")));
    }
    if has("const") && has("enum") {
        return Err(invalid(format!("{path}: use either 'const' or 'enum', not both")));
    }
    let values = object
        .get("const")
        .map(std::slice::from_ref)
        .or_else(|| object.get("enum").and_then(Value::as_array).map(Vec::as_slice));
    if let (Some(values), Some(kind)) = (values, object.get("type"))
        && let Some(value) = values.iter().find(|value| !allows(kind, value))
    {
        return Err(invalid(format!("{path}: enum value {value} does not match 'type'")));
    }
    let numeric = object.get("type").is_some_and(|kind| allows(kind, &Value::from(0)));
    if (has("minimum") || has("maximum")) && (values.is_some() || !numeric) {
        return Err(invalid(format!(
            "{path}: 'minimum' and 'maximum' apply only to an integer scale"
        )));
    }
    Ok(())
}

fn unsupported(prefix: &str, object: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    object
        .keys()
        .find(|key| !allowed.contains(&key.as_str()) && !ANNOTATIONS.contains(&key.as_str()))
        .map_or(Ok(()), |key| {
            Err(invalid(format!("{prefix}unsupported schema keyword '{key}'")))
        })
}

/// Whether a JSON Schema `type` (a name or a list of names) admits `value`.
fn allows(kind: &Value, value: &Value) -> bool {
    match kind {
        Value::String(name) => match name.as_str() {
            "null" => value.is_null(),
            "boolean" => value.is_boolean(),
            "integer" => value.as_f64().is_some_and(|number| number.fract() == 0.0),
            "number" => value.is_number(),
            "string" => value.is_string(),
            _ => false,
        },
        Value::Array(kinds) => kinds.iter().any(|kind| allows(kind, value)),
        _ => false,
    }
}
