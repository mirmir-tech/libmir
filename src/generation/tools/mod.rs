mod diagnostic;
#[cfg(test)]
mod tests;
mod xml;

use foundation::conversation::{Conversation, ToolCall, ToolChoice};

use crate::{Error, Result};

pub(super) fn normalize(payload: &str, conversation: &Conversation) -> Result<String> {
    if payload.trim().is_empty() {
        return match conversation.tool_choice {
            ToolChoice::Required | ToolChoice::Function(_) => {
                Err(invalid("required tool call was not emitted"))
            },
            ToolChoice::Auto | ToolChoice::None => Ok(String::new()),
        };
    }
    let calls = if payload.trim_start().starts_with("<tool_call>") {
        xml::parse(payload, &conversation.tools)?
    } else {
        ToolCall::parse_mistral(payload).map_err(Error::InvalidToolCall)?
    };
    if calls.is_empty() {
        return Err(invalid("tool-call output is empty"));
    }
    for call in &calls {
        match &conversation.tool_choice {
            ToolChoice::None => return Err(invalid("tool calls were disabled")),
            ToolChoice::Function(name) if *name != call.function.name => {
                return Err(invalid("model called a different tool than requested"));
            },
            _ => {},
        }
        if !conversation.tools.iter().any(|tool| tool.function.name == call.function.name) {
            return Err(invalid("model called an undeclared tool"));
        }
    }
    serde_json::to_string(&calls).map_err(|error| invalid(error.to_string()))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidToolCall(message.into())
}
