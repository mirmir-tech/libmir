use foundation::conversation::{Conversation, ToolChoice};

use super::ReasoningMode;
use crate::{ModelsError, error::Result};

/// A native tool-call header already included in the model's input tokens.
#[derive(Debug, Clone)]
pub enum ToolCallPrefix {
    XmlFunction(String),
}

impl ToolCallPrefix {
    #[must_use]
    pub fn text(&self) -> String {
        match self {
            Self::XmlFunction(name) => format!("<tool_call>\n<function={name}>\n"),
        }
    }
}

pub(super) fn prefix(
    template: Option<&str>,
    conversation: &Conversation,
    reasoning: ReasoningMode,
) -> Result<Option<ToolCallPrefix>> {
    let ToolChoice::Function(name) = &conversation.tool_choice else {
        return Ok(None);
    };
    if !conversation.tools.iter().any(|tool| tool.function.name == *name) {
        return Err(ModelsError::InvalidConfig("named tool is absent from tools".into()));
    }
    // Preserve explicit/default reasoning. Only start the named function once
    // the no-thinking assistant prefix has already closed the reasoning block.
    if reasoning != ReasoningMode::Disabled
        || !template.is_some_and(|body| body.contains("<tool_call>") && body.contains("<function="))
    {
        return Ok(None);
    }
    if name.is_empty()
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'))
    {
        return Err(ModelsError::InvalidConfig("named XML tool has an invalid name".into()));
    }
    Ok(Some(ToolCallPrefix::XmlFunction(name.clone())))
}
