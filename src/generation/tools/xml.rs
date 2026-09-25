use foundation::conversation::{FunctionCall, Tool, ToolCall};
use serde_json::{Map, Value};

use super::{diagnostic::json_error, invalid};
use crate::Result;

pub(super) fn parse(mut input: &str, tools: &[Tool]) -> Result<Vec<ToolCall>> {
    let mut calls = Vec::new();
    while !input.trim().is_empty() {
        input = strip(input, "<tool_call>")?;
        let (body, remainder) = input
            .split_once("</tool_call>")
            .ok_or_else(|| invalid("unterminated tool_call"))?;
        let function = if body.trim_start().starts_with('{') {
            serde_json::from_str::<FunctionCall>(body)
                .map_err(|error| json_error("invalid tool JSON", body, &error))?
        } else {
            function(body, tools)?
        };
        calls.push(ToolCall {
            id: format!("call{:05}", calls.len()),
            kind: "function".into(),
            function,
        });
        input = remainder;
    }
    Ok(calls)
}

fn function(input: &str, tools: &[Tool]) -> Result<FunctionCall> {
    let input = strip(input, "<function=")?;
    let (name, mut input) =
        input.split_once('>').ok_or_else(|| invalid("unterminated function name"))?;
    let tool = tools
        .iter()
        .find(|tool| tool.function.name == name)
        .ok_or_else(|| invalid("unknown XML tool name"))?;
    let properties = tool.function.parameters.get("properties");
    let mut arguments = Map::new();
    loop {
        if input.trim_start().starts_with("</function>") {
            if !strip(input, "</function>")?.trim().is_empty() {
                return Err(invalid("unexpected text after function"));
            }
            break;
        }
        input = strip(input, "<parameter=")?;
        let (name, body) =
            input.split_once('>').ok_or_else(|| invalid("unterminated parameter name"))?;
        let (value, remainder) = body
            .split_once("</parameter>")
            .ok_or_else(|| invalid("unterminated parameter"))?;
        let schema = properties
            .and_then(|properties| properties.get(name))
            .ok_or_else(|| invalid("unknown XML parameter"))?;
        let value = argument(value.trim(), schema)?;
        if arguments.insert(name.to_owned(), value).is_some() {
            return Err(invalid("duplicate XML parameter"));
        }
        input = remainder;
    }
    if let Some(required) = tool.function.parameters.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            if !arguments.contains_key(name) {
                return Err(invalid(format!("missing required parameter {name}")));
            }
        }
    }
    Ok(FunctionCall {
        name: name.into(),
        arguments: Value::Object(arguments),
    })
}

fn argument(input: &str, schema: &Value) -> Result<Value> {
    let kind = schema.get("type");
    if kind.and_then(Value::as_str) == Some("string") {
        return Ok(serde_json::from_str::<String>(input)
            .map_or_else(|_| Value::String(input.into()), Value::String));
    }
    let parsed: Value = serde_json::from_str(input)
        .map_err(|error| json_error("non-string parameter must contain JSON", input, &error))?;
    let accepts = |kind: &str| match kind {
        "null" => parsed.is_null(),
        "string" => parsed.is_string(),
        "number" => parsed.is_number(),
        "integer" => parsed.is_i64() || parsed.is_u64(),
        "boolean" => parsed.is_boolean(),
        "array" => parsed.is_array(),
        "object" => parsed.is_object(),
        _ => false,
    };
    let valid = match kind {
        Some(Value::String(kind)) => accepts(kind),
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).any(accepts),
        None => true,
        _ => false,
    };
    if !valid {
        return Err(invalid("XML argument does not match its declared type"));
    }
    Ok(parsed)
}

fn strip<'a>(input: &'a str, prefix: &str) -> Result<&'a str> {
    input
        .trim_start()
        .strip_prefix(prefix)
        .ok_or_else(|| invalid(format!("expected {prefix}")))
}
