use std::fmt::Write;

use llguidance::JsonCompileOptions;
use serde_json::{Value, json};

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
    let mut rules = String::from("_ws: /[ \\t\\r\\n]{0,16}/\n");
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
        // Bound formatting gaps at both grammar layers. Unbounded envelope or
        // JSON skip whitespace lets greedy decoding spend the entire generation
        // budget on indentation before a required quoted parameter value.
        // Whitespace within a JSON string remains ordinary, unconstrained data.
        let mut value = if value.is_object() {
            value.clone()
        } else {
            json!({"allOf":[value]})
        };
        JsonCompileOptions {
            whitespace_pattern: Some(r"[ \t\r\n]{1,16}".into()),
            ..JsonCompileOptions::default()
        }
        .apply_to(&mut value);
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

/// Add a named native envelope when it was not prefilled before reasoning.
pub(super) fn with_prefix(grammar: &str, prefix: &models::chat::ToolCallPrefix) -> Result<String> {
    let literal =
        serde_json::to_string(&prefix.text()).map_err(|error| invalid(error.to_string()))?;
    Ok(grammar.replacen("start: _ws", &format!("start: _ws {literal} _ws"), 1))
}
