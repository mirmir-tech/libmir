use std::borrow::Cow;

use foundation::conversation::{Conversation, Message};

use crate::generation::ToolConstraints;

// Native XML templates demonstrate unquoted string parameters. Our grammar
// constrains every parameter as JSON, so the rendered prompt must explain that
// difference before sampling. Otherwise masking can turn a raw-text
// continuation into whitespace loops or schema-valid but meaningless strings.
const FORMAT: &str = "SCHEMA CONSTRAINED TOOL FORMAT: Inside every <parameter=name> tag, write exactly one JSON value. String parameters must include JSON double quotes, for example <parameter=message>\"The requested text.\"</parameter>. Arrays and objects use JSON normally. Do not write raw unquoted strings.";

pub(in crate::generation) fn conversation(
    original: &Conversation,
    constraints: ToolConstraints,
) -> Cow<'_, Conversation> {
    if constraints == ToolConstraints::None {
        return Cow::Borrowed(original);
    }
    let mut conversation = original.clone();
    if let Some(system) = conversation.messages.first_mut().filter(|m| m.role == "system") {
        system.content.push_str("\n\n");
        system.content.push_str(FORMAT);
    } else {
        conversation.messages.insert(
            0,
            Message {
                role: "system".into(),
                content: FORMAT.into(),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }
    Cow::Owned(conversation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn schema_format_preserves_original_instructions_without_mutating_request() {
        let original = Conversation {
            messages: vec![message("system", "Keep source facts."), message("user", "Evidence")],
            ..Conversation::default()
        };
        let prepared = conversation(&original, ToolConstraints::Schema);
        assert_eq!(prepared.messages.len(), 2);
        assert_eq!(prepared.messages[0].content, format!("Keep source facts.\n\n{FORMAT}"));
        assert_eq!(prepared.messages[1].content, "Evidence");
        assert_eq!(original.messages[0].content, "Keep source facts.");
        assert!(matches!(conversation(&original, ToolConstraints::None), Cow::Borrowed(_)));
    }

    #[test]
    fn schema_format_adds_system_when_caller_has_none() {
        let original = Conversation {
            messages: vec![message("user", "Evidence")],
            ..Conversation::default()
        };
        let prepared = conversation(&original, ToolConstraints::Schema);
        assert_eq!(prepared.messages.len(), 2);
        assert_eq!(prepared.messages[0].role, "system");
        assert_eq!(prepared.messages[0].content, FORMAT);
        assert_eq!(prepared.messages[1].content, "Evidence");
    }
}
