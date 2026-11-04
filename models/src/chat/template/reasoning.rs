use serde::{Deserialize, Serialize};

/// Request-scoped control of the model's declared thinking template switch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    /// Preserve the runtime's existing model-template behavior.
    #[default]
    ModelDefault,
    /// Enable thinking when the model declares a supported template switch.
    Enabled,
    /// Disable thinking when the model declares a supported template switch.
    Disabled,
}

impl ReasoningMode {
    pub(super) const fn thinking_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

/// Request-scoped reasoning depth, for templates that declare a switch for
/// shorter reasoning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Keep the template's default depth.
    #[default]
    ModelDefault,
    /// Ask the model for shorter, more efficient reasoning.
    Efficient,
}

/// Whether earlier assistant turns keep their reasoning in the prompt, for
/// templates that declare the choice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningHistory {
    /// Keep the template's default.
    #[default]
    ModelDefault,
    /// Render earlier turns with their reasoning.
    Keep,
    /// Drop the reasoning of turns before the last user message.
    Truncate,
}

/// Template-level reasoning controls of one request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TemplateControls {
    /// The thinking switch.
    pub reasoning: ReasoningMode,
    /// The reasoning depth.
    pub effort: ReasoningEffort,
    /// Reasoning of earlier turns.
    pub history: ReasoningHistory,
}

impl From<ReasoningMode> for TemplateControls {
    fn from(reasoning: ReasoningMode) -> Self {
        Self { reasoning, ..Self::default() }
    }
}
