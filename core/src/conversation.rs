use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(deserialize_with = "json_or_encoded_json")]
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Parses the native Mistral V3 `[TOOL_CALLS]` JSON payload.
    pub fn parse_mistral(payload: &str) -> Result<Vec<Self>, String> {
        let values = serde_json::from_str::<Vec<serde_json::Value>>(payload)
            .map_err(|error| format!("invalid tool-call JSON: {error}"))?;
        values
            .into_iter()
            .enumerate()
            .map(|(index, value)| Self::from_mistral_value(value, index))
            .collect()
    }

    fn from_mistral_value(mut value: serde_json::Value, index: usize) -> Result<Self, String> {
        let object = value.as_object_mut().ok_or("tool call must be an object")?;
        let id = object
            .remove("id")
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| format!("call{:05}", index % 100_000));
        let kind = object
            .remove("type")
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(function_kind);
        let function = match object.remove("function") {
            Some(value) => serde_json::from_value(value)
                .map_err(|error| format!("invalid tool-call function: {error}"))?,
            None => FunctionCall {
                name: take_string(object, "name")?,
                arguments: decode_json_string(
                    object.remove("arguments").unwrap_or(serde_json::Value::Null),
                ),
            },
        };
        Ok(Self { id, kind, function })
    }
}

fn take_string(
    object: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<String, String> {
    object
        .remove(key)
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| format!("tool call is missing string `{key}`"))
}

fn json_or_encoded_json<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(decode_json_string(value))
}

fn decode_json_string(value: serde_json::Value) -> serde_json::Value {
    if let serde_json::Value::String(encoded) = &value {
        serde_json::from_str(encoded).unwrap_or(value)
    } else {
        value
    }
}

fn function_kind() -> String {
    "function".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_native_mistral_tool_calls() -> Result<(), String> {
        let calls = ToolCall::parse_mistral(
            r#"[{"name":"weather","arguments":{"city":"Warsaw"},"id":"abc123456"}]"#,
        )?;
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments["city"], "Warsaw");
        Ok(())
    }

    #[test]
    fn assigns_mistral_compatible_id_when_model_omits_it() -> Result<(), String> {
        let calls = ToolCall::parse_mistral(r#"[{"name":"weather","arguments":{}}]"#)?;
        assert_eq!(calls[0].id, "call00000");
        Ok(())
    }
}
