use foundation::conversation::{FunctionCall, Tool, ToolCall};
use serde_json::{Map, Value};

use super::{diagnostic::json_error, invalid};
use crate::Result;

pub(super) fn parse(mut input: &str, tools: &[Tool]) -> Result<Vec<ToolCall>> {
    let mut calls = Vec::new();
    while !input.trim().is_empty() {
        input = strip(input, "<tool_call>")?;
        let (function, remainder) = if input.trim_start().starts_with('{') {
            let input = input.trim_start();
            let mut values = serde_json::Deserializer::from_str(input).into_iter::<FunctionCall>();
            let function = values
                .next()
                .ok_or_else(|| invalid("missing tool JSON"))?
                .map_err(|error| json_error("invalid tool JSON", input, &error))?;
            (function, &input[values.byte_offset()..])
        } else {
            function(input, tools)?
        };
        let remainder = strip(remainder, "</tool_call>")?;
        calls.push(ToolCall {
            id: format!("call{:05}", calls.len()),
            kind: "function".into(),
            function,
        });
        input = remainder;
    }
    Ok(calls)
}

fn function<'a>(input: &'a str, tools: &[Tool]) -> Result<(FunctionCall, &'a str)> {
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
            input = strip(input, "</function>")?;
            break;
        }
        input = strip(input, "<parameter=")?;
        let (name, body) =
            input.split_once('>').ok_or_else(|| invalid("unterminated parameter name"))?;
        let schema = properties
            .and_then(|properties| properties.get(name))
            .ok_or_else(|| invalid("unknown XML parameter"))?;
        let body = body.trim_start();
        let mut values = serde_json::Deserializer::from_str(body).into_iter::<Value>();
        // Structured values may contain literal XML tags inside JSON strings.
        // Only a complete JSON value followed by the delimiter owns that delimiter.
        let parsed = values.next().and_then(std::result::Result::ok);
        let boundary = values.byte_offset();
        let (value, remainder) =
            if parsed.is_some() && body[boundary..].trim_start().starts_with("</parameter>") {
                (&body[..boundary], strip(&body[boundary..], "</parameter>")?)
            } else {
                body.split_once("</parameter>")
                    .ok_or_else(|| invalid("unterminated parameter"))?
            };
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
    Ok((
        FunctionCall {
            name: name.into(),
            arguments: Value::Object(arguments),
        },
        input,
    ))
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
