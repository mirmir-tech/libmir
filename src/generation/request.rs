use foundation::conversation::Conversation;
use models::generation::GenerationOverrides;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Request-scoped handling for a detected reasoning cycle.
pub enum ReasoningCyclePolicy {
    /// Resume greedy decoding with a repetition penalty.
    Recover,
    /// Close a supported reasoning channel after the configured token gate.
    ExitReasoning {
        /// Earliest generated-token position eligible for cycle detection.
        min_tokens: usize,
    },
}

impl Default for ReasoningCyclePolicy {
    fn default() -> Self {
        Self::ExitReasoning { min_tokens: 64 }
    }
}

#[derive(Debug, Clone, Default)]
/// Backend-neutral input for one model generation.
pub struct GenerationRequest {
    /// Ordered conversation turns and callable tools rendered by the model
    /// template.
    pub conversation: Conversation,
    /// Request-scoped sampling values applied over the loaded model defaults.
    pub options: GenerationOverrides,
    /// Optional deterministic sampler seed.
    pub seed: Option<u64>,
    /// Policy applied after a high-confidence reasoning cycle is detected.
    pub reasoning_cycle: ReasoningCyclePolicy,
    /// Controls the prompt thinking mode; output limits still count all
    /// channels.
    pub reasoning: crate::ReasoningMode,
    /// Opt-in schema-constrained native tool decoding.
    pub tool_constraints: ToolConstraints,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_exit_is_default() {
        assert_eq!(
            ReasoningCyclePolicy::default(),
            ReasoningCyclePolicy::ExitReasoning { min_tokens: 64 }
        );
    }
}

/// Explicit request-scoped native tool output constraints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolConstraints {
    /// Preserve ordinary model decoding.
    #[default]
    None,
    /// Enforce the named tool's schema while sampling its native XML arguments.
    Schema,
}
