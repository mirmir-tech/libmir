use serde_json::Value;

use super::{Projection, SchemaField, invalid};
use crate::{
    decision::{ChoiceOption, Question, Verdicts},
    error::Result,
};

const MAX_OPTIONS: usize = 32;
const MAX_SCORE_LEVELS: i64 = 10;

/// Plans one property; `inherited` is the description of an enclosing
/// `Optional[...]` union.
pub(super) fn plan(
    path: &str,
    name: &str,
    property: &Value,
    inherited: Option<&str>,
) -> Result<SchemaField> {
    let object = property
        .as_object()
        .ok_or_else(|| invalid(format!("{path}: property must be an object")))?;
    let description = object.get("description").and_then(Value::as_str).or(inherited);
    if !["const", "enum", "type"].iter().any(|key| object.contains_key(*key))
        && let Some(union) =
            object.get("anyOf").or_else(|| object.get("oneOf")).and_then(Value::as_array)
    {
        let branches: Vec<&Value> = union
            .iter()
            .filter(|branch| branch.get("type") != Some(&Value::from("null")))
            .collect();
        let [branch] = branches.as_slice() else {
            return Err(invalid(format!(
                "{path}: only unions with one non-null branch are supported"
            )));
        };
        return plan(path, name, branch, description);
    }
    if let Some(value) = object.get("const") {
        return enumeration(path, name, std::slice::from_ref(value), description);
    }
    if let Some(values) = object.get("enum") {
        let values = values
            .as_array()
            .ok_or_else(|| invalid(format!("{path}: 'enum' must be an array")))?;
        return enumeration(path, name, values, description);
    }
    match json_type(path, object.get("type"))? {
        Some("boolean") => boolean(name, description),
        Some("integer" | "number") => score(path, name, object, description),
        Some("string") => Err(invalid(format!(
            "{path}: a free string has no fixed option set; use 'enum' or a boolean"
        ))),
        Some("array") => {
            Err(invalid(format!("{path}: arrays are not supported; ask one field per element")))
        },
        Some("object") => {
            Err(invalid(format!("{path}: nested objects are not supported; flatten the schema")))
        },
        _ => Err(invalid(format!("{path}: unsupported schema"))),
    }
}

fn json_type<'a>(path: &str, kind: Option<&'a Value>) -> Result<Option<&'a str>> {
    match kind {
        Some(Value::String(kind)) => Ok(Some(kind)),
        Some(Value::Array(kinds)) => {
            let non_null: Vec<&str> =
                kinds.iter().filter_map(Value::as_str).filter(|kind| *kind != "null").collect();
            match non_null.as_slice() {
                [kind] => Ok(Some(kind)),
                [] => Ok(None),
                _ => Err(invalid(format!("{path}: 'type' has several non-null types"))),
            }
        },
        _ => Ok(None),
    }
}

fn enumeration(
    path: &str,
    name: &str,
    values: &[Value],
    description: Option<&str>,
) -> Result<SchemaField> {
    if values.is_empty() || values.len() > MAX_OPTIONS {
        return Err(invalid(format!("{path}: 'enum' needs 1 to {MAX_OPTIONS} values")));
    }
    if values.iter().all(Value::is_boolean) {
        return boolean(name, description);
    }
    let options = values
        .iter()
        .map(|value| Ok(ChoiceOption::new(label(path, value)?, None)))
        .collect::<Result<Vec<_>>>()?;
    let instruction = description.map_or_else(|| format!("What is `{name}`?"), str::to_owned);
    Ok(SchemaField {
        name: name.into(),
        question: Question::choice(instruction, options)?,
        projection: Projection::Choice(values.to_vec()),
    })
}

/// The option text Python's `str()` gives an enum value.
fn label(path: &str, value: &Value) -> Result<String> {
    match value {
        Value::Null => Ok("null".into()),
        Value::Bool(true) => Ok("True".into()),
        Value::Bool(false) => Ok("False".into()),
        Value::Number(number) => Ok(number.to_string()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(_) | Value::Object(_) => {
            Err(invalid(format!("{path}: enum values must be scalars")))
        },
    }
}

fn boolean(name: &str, description: Option<&str>) -> Result<SchemaField> {
    let instruction = description.map_or_else(|| format!("Is `{name}` true?"), str::to_owned);
    Ok(SchemaField {
        name: name.into(),
        question: Question::yes_no(instruction, Verdicts::default())?,
        projection: Projection::Boolean,
    })
}

fn score(
    path: &str,
    name: &str,
    object: &serde_json::Map<String, Value>,
    description: Option<&str>,
) -> Result<SchemaField> {
    let bound = |key: &str| object.get(key).and_then(Value::as_i64);
    let (Some(minimum), Some(maximum)) = (bound("minimum"), bound("maximum")) else {
        return Err(invalid(format!(
            "{path}: a numeric field needs integer 'minimum' and 'maximum'"
        )));
    };
    if maximum < minimum || maximum - minimum + 1 > MAX_SCORE_LEVELS {
        return Err(invalid(format!("{path}: a score needs 1 to {MAX_SCORE_LEVELS} levels")));
    }
    let levels = (minimum..=maximum).map(|level| level.to_string()).collect();
    let instruction = description
        .map_or_else(|| format!("Score `{name}` from {minimum} to {maximum}"), str::to_owned);
    Ok(SchemaField {
        name: name.into(),
        question: Question::score(instruction, levels)?,
        projection: Projection::Score { minimum },
    })
}
