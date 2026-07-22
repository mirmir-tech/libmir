use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ChatTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, deserialize_with = "string_or_default")]
    pub content: String,
    #[serde(default, alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTool {
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: ChatFunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: ChatFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFunctionCall {
    pub name: String,
    #[serde(deserialize_with = "json_or_encoded_json")]
    pub arguments: serde_json::Value,
}

impl ChatToolCall {
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
            None => ChatFunctionCall {
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

fn string_or_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirmir: Option<MirmirChatInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MirmirChatInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStreamChunk {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStreamChoice {
    pub index: usize,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_native_mistral_tool_calls() -> Result<(), String> {
        let calls = ChatToolCall::parse_mistral(
            r#"[{"name":"weather","arguments":{"city":"Warsaw"},"id":"abc123456"}]"#,
        )?;

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "abc123456");
        assert_eq!(calls[0].kind, "function");
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments["city"], "Warsaw");
        Ok(())
    }

    #[test]
    fn assigns_mistral_compatible_id_when_model_omits_it() -> Result<(), String> {
        let calls =
            ChatToolCall::parse_mistral(r#"[{"name":"weather","arguments":{"city":"Warsaw"}}]"#)?;

        assert_eq!(calls[0].id, "call00000");
        Ok(())
    }
}
