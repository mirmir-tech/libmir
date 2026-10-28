use super::invalid;
use crate::error::Result;

const DEFAULT_FALSE: &str = "no, the statement does not hold";
const DEFAULT_TRUE: &str = "yes, the statement holds";

/// A labelled choice option with an optional description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChoiceOption {
    pub label: String,
    pub description: Option<String>,
}

impl ChoiceOption {
    #[must_use]
    pub fn new(label: impl Into<String>, description: Option<String>) -> Self {
        Self { label: label.into(), description }
    }

    pub(super) fn render(&self) -> String {
        self.description.as_deref().filter(|text| !text.is_empty()).map_or_else(
            || self.label.clone(),
            |description| format!("{}: {description}", self.label),
        )
    }
}

/// Labels and optional descriptions of the two answers to a yes/no
/// question; the false answer always comes first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdicts {
    false_label: String,
    true_label: String,
    false_description: Option<String>,
    true_description: Option<String>,
}

impl Default for Verdicts {
    fn default() -> Self {
        Self {
            false_label: "false".into(),
            true_label: "true".into(),
            false_description: None,
            true_description: None,
        }
    }
}

impl Verdicts {
    /// Replaces the default `false`/`true` labels.
    pub fn with_labels(mut self, false_label: &str, true_label: &str) -> Result<Self> {
        let (false_label, true_label) = (false_label.trim(), true_label.trim());
        if false_label.is_empty() || true_label.is_empty() || false_label == true_label {
            return Err(invalid("yes/no labels must be distinct and non-empty"));
        }
        self.false_label = false_label.into();
        self.true_label = true_label.into();
        Ok(self)
    }

    /// Describes when the statement does not hold and when it does.
    #[must_use]
    pub fn with_descriptions(
        mut self,
        false_description: Option<String>,
        true_description: Option<String>,
    ) -> Self {
        self.false_description = false_description;
        self.true_description = true_description;
        self
    }

    pub(super) fn render(&self) -> [String; 2] {
        let describe = |description: &Option<String>, fallback: &str| {
            description
                .as_deref()
                .filter(|text| !text.is_empty())
                .unwrap_or(fallback)
                .to_owned()
        };
        [
            format!("{}: {}", self.false_label, describe(&self.false_description, DEFAULT_FALSE)),
            format!("{}: {}", self.true_label, describe(&self.true_description, DEFAULT_TRUE)),
        ]
    }
}
