use foundation::conversation::Conversation;
use models::generation::GenerationOverrides;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Request-scoped handling for a detected reasoning cycle.
pub enum ReasoningCyclePolicy {
    /// Resume greedy decoding with a repetition penalty.
    Recover,
    /// Close a supported reasoning channel after three consecutive token cycles
    /// and the configured token gate. Scattered repeated quotations do not
    /// trigger an exit. Completion limits still bound unfinished reasoning.
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
    /// Requests shorter reasoning from templates that declare the switch.
    pub reasoning_effort: crate::ReasoningEffort,
    /// Keeps or drops earlier turns' reasoning in templates that declare
    /// the choice.
    pub reasoning_history: crate::ReasoningHistory,
    /// Opt-in schema-constrained native tool decoding.
    pub tool_constraints: ToolConstraints,
    /// Shape of the final content; a JSON schema constrains it like the
    /// arguments of a schema tool, without any tool-call syntax.
    pub response_format: ResponseFormat,
    /// Optional reasoning-token cap for enabled schema-constrained XML tools.
    /// The native delimiter is forced after this many reasoning tokens; the
    /// total completion budget still includes the delimiter and tool output.
    pub reasoning_token_budget: Option<std::num::NonZeroUsize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_content_rejects_forced_tools_but_keeps_optional_tools() -> crate::Result<()> {
        use foundation::conversation::ToolChoice;
        let json = ResponseFormat::JsonSchema(serde_json::json!({"type":"object"}));
        for choice in [ToolChoice::Required, ToolChoice::Function("emit".into())] {
            assert!(json.validate_tool_choice(&choice).is_err());
            ResponseFormat::Text.validate_tool_choice(&choice)?;
        }
        for choice in [ToolChoice::Auto, ToolChoice::None] {
            json.validate_tool_choice(&choice)?;
        }
        Ok(())
    }

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

/// Request-scoped shape of the final assistant content.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Ordinary text.
    #[default]
    Text,
    /// One JSON value matching the schema, decoded under its grammar and
    /// validated before it is returned.
    JsonSchema(serde_json::Value),
}

impl ResponseFormat {
    /// JSON content cannot simultaneously satisfy a required tool call.
    /// Optional tools may remain in the conversation as prompt context.
    pub fn validate_tool_choice(
        &self,
        choice: &foundation::conversation::ToolChoice,
    ) -> crate::Result<()> {
        use foundation::conversation::ToolChoice;
        if matches!(self, Self::JsonSchema(_))
            && matches!(choice, ToolChoice::Required | ToolChoice::Function(_))
        {
            return Err(models::ModelsError::InvalidConfig(
                "a JSON response_format cannot be combined with a required or named tool_choice"
                    .into(),
            )
            .into());
        }
        Ok(())
    }
}
