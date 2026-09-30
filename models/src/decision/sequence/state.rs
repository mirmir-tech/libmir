use std::io;

use serde::Serialize;
use serde_json::ser::Formatter;

use crate::error::Result;

/// Which end of an over-long state survives truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateEnd {
    /// Keep the beginning, as for a document.
    Start,
    /// Keep the most recent part, as for a chronological conversation.
    End,
}

/// The evidence every question of one decision is asked about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionState {
    text: String,
    kept: StateEnd,
}

impl DecisionState {
    /// Plain text; truncation keeps its beginning.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: text.into(), kept: StateEnd::Start }
    }

    /// A structured document, serialized the way the reference runtime does;
    /// truncation keeps its beginning.
    pub fn document(value: &impl Serialize) -> Result<Self> {
        Ok(Self {
            text: python_json(value)?,
            kept: StateEnd::Start,
        })
    }

    /// Conversation turns, oldest first, serialized as a JSON list the way
    /// the reference runtime does; truncation keeps the newest turns.
    pub fn conversation<T: Serialize>(turns: &[T]) -> Result<Self> {
        Ok(Self {
            text: python_json(&turns)?,
            kept: StateEnd::End,
        })
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    #[must_use]
    pub const fn kept(&self) -> StateEnd {
        self.kept
    }
}

/// Serializes like Python's `json.dumps(value, ensure_ascii=False)`: `", "`
/// between items, `": "` after keys, non-ASCII text unescaped, and fields in
/// their declaration order.
fn python_json(value: &impl Serialize) -> Result<String> {
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, PythonFormatter);
    value.serialize(&mut serializer)?;
    Ok(std::str::from_utf8(&bytes)?.to_owned())
}

struct PythonFormatter;

impl Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

#[cfg(test)]
mod tests {
    use serde::Serialize;

    use super::{DecisionState, StateEnd};
    use crate::Result;

    #[derive(Serialize)]
    struct Turn<'a> {
        speaker: &'a str,
        text: &'a str,
    }

    #[test]
    fn serializes_conversations_like_python_json() -> Result<()> {
        let state = DecisionState::conversation(&[
            Turn {
                speaker: "Ania",
                text: "Czy agent już skończył?",
            },
            Turn {
                speaker: "agent",
                text: "Tak.\n\"Gotowe\"",
            },
        ])?;
        assert_eq!(
            state.as_str(),
            r#"[{"speaker": "Ania", "text": "Czy agent już skończył?"}, {"speaker": "agent", "text": "Tak.\n\"Gotowe\""}]"#
        );
        assert_eq!(state.kept(), StateEnd::End);
        assert_eq!(DecisionState::text("x").kept(), StateEnd::Start);
        Ok(())
    }
}
