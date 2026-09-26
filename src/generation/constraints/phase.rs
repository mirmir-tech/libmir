use models::chat::ToolCallPrefix;

use super::{GenerationRequest, PreparedGeneration, invalid};
use crate::{Model, Result};

pub(super) enum Phase {
    Reasoning { end: u32 },
    Tool,
}

impl Phase {
    pub(super) fn prepare(
        model: &Model,
        request: &GenerationRequest,
        prepared: &PreparedGeneration,
    ) -> Result<(ToolCallPrefix, Self)> {
        if let Some(prefix) = prepared.tool_prefix() {
            return Ok((prefix.clone(), Self::Tool));
        }
        let descriptor = model.descriptor();
        let prefix = descriptor
            .template()
            .named_tool_prefix(&request.conversation)?
            .ok_or_else(|| invalid("schema mode requires an explicitly named XML tool"))?;
        let end = descriptor
            .tokenizer()
            .xml_reasoning_end_token(prepared.prompt_text())
            .ok_or_else(|| {
                invalid("schema reasoning requires a native think prefix and atomic closing token")
            })?;
        Ok((prefix, Self::Reasoning { end }))
    }

    pub(super) fn is_reasoning(&self) -> bool {
        matches!(self, Self::Reasoning { .. })
    }

    pub(super) fn end_token(&self) -> Option<u32> {
        match self {
            Self::Reasoning { end } => Some(*end),
            Self::Tool => None,
        }
    }

    /// Reasoning tokens, including the transition delimiter, are never fed to
    /// the tool grammar. From the next token onward its mask is authoritative.
    pub(super) fn observe(&mut self, token: u32) -> bool {
        let Self::Reasoning { end } = self else {
            return false;
        };
        if token == *end {
            *self = Self::Tool;
        }
        true
    }
}
