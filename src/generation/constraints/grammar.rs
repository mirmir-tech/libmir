use std::fmt::Write;

use serde_json::Value;

use super::invalid;
use crate::Result;

/// Keep the model's native envelope; each parameter value is schema-bound JSON.
/// Parameter order is canonical and optional properties may be omitted.
pub(super) fn compile(schema: &Value, tool_end: Option<u32>) -> Result<String> {
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err(invalid("tool parameters must be an object schema"));
    }
    let root = schema.as_object().ok_or_else(|| invalid("invalid tool schema"))?;
    if root.keys().any(|key| {
        !matches!(
            key.as_str(),
            "type"
                | "properties"
                | "required"
                | "additionalProperties"
                | "$defs"
                | "description"
                | "title"
        )
    }) {
        return Err(invalid("unsupported constraint at tool parameter root"));
    }
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("tool schema has no properties"))?;
    if properties.len() > 64 || schema.to_string().len() > 512_000 {
        return Err(invalid("tool schema exceeds constraint limits"));
    }
    let required = match schema.get("required") {
        Some(Value::Array(values)) => values.clone(),
        None => Vec::new(),
        Some(_) => return Err(invalid("required must be an array")),
    };
    if required
        .iter()
        .any(|key| key.as_str().is_none_or(|key| !properties.contains_key(key)))
    {
        return Err(invalid("required tool parameter is undeclared"));
    }
    let mut start = String::from("start: _ws");
    let mut rules = String::from("_ws: /[ \\t\\r\\n]*/\n");
    for (index, (name, value)) in properties.iter().enumerate() {
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        {
            return Err(invalid("invalid XML parameter name"));
        }
        let optional = if required.iter().any(|key| key.as_str() == Some(name)) {
            ""
        } else {
            "?"
        };
        write!(start, " param{index}{optional}").map_err(|error| invalid(error.to_string()))?;
        let mut value = value.clone();
        if let (Some(definitions), Some(object)) = (schema.get("$defs"), value.as_object_mut()) {
            object.insert("$defs".into(), definitions.clone());
        }
        let open = serde_json::to_string(&format!("<parameter={name}>"))
            .map_err(|error| invalid(error.to_string()))?;
        let close =
            serde_json::to_string("</parameter>").map_err(|error| invalid(error.to_string()))?;
        write!(
            rules,
            "param{index}: {open} _ws value{index} _ws {close} _ws\nvalue{index}: %json {value}\n"
        )
        .map_err(|error| invalid(error.to_string()))?;
    }
    let close = tool_end.map_or_else(
        || "\"</function>\" _ws \"</tool_call>\"".into(),
        |token| format!("\"</function>\" _ws <[{token}]>"),
    );
    writeln!(start, " {close}").map_err(|error| invalid(error.to_string()))?;
    Ok(format!("{start}{rules}"))
}
