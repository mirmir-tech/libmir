mod diagnostic;
#[cfg(test)]
mod tests;
mod xml;

use foundation::conversation::{Conversation, ToolCall, ToolChoice};

use crate::{Error, Result, ToolCallError};

pub(super) fn normalize(payload: &str, conversation: &Conversation) -> Result<String> {
    if payload.trim().is_empty() {
        return match conversation.tool_choice {
            ToolChoice::Required | ToolChoice::Function(_) => {
                Err(ToolCallError::MissingRequired.into())
            },
            ToolChoice::Auto | ToolChoice::None => Ok(String::new()),
        };
    }
    let calls = if payload.trim_start().starts_with("<tool_call>") {
        xml::parse(payload, &conversation.tools)?
    } else {
        ToolCall::parse_mistral(payload).map_err(ToolCallError::Malformed)?
    };
    if calls.is_empty() {
        return Err(ToolCallError::Empty.into());
    }
    for call in &calls {
        match &conversation.tool_choice {
            ToolChoice::None => return Err(ToolCallError::Disabled.into()),
            ToolChoice::Function(name) if *name != call.function.name => {
                return Err(ToolCallError::WrongFunction {
                    expected: name.clone(),
                    actual: call.function.name.clone(),
                }
                .into());
            },
            _ => {},
        }
        if !conversation.tools.iter().any(|tool| tool.function.name == call.function.name) {
            return Err(ToolCallError::UndeclaredFunction(call.function.name.clone()).into());
        }
    }
    serde_json::to_string(&calls)
        .map_err(|error| Error::from(ToolCallError::Malformed(error.to_string())))
}

fn invalid(message: impl Into<String>) -> Error {
    ToolCallError::Malformed(message.into()).into()
}
