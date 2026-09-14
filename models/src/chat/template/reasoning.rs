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
