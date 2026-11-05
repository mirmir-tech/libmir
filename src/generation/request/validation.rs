use super::{GenerationRequest, ResponseFormat, ToolConstraints};
use crate::{Result, generation::constraints::invalid};

impl GenerationRequest {
    /// Checks model-independent output constraints before starting work or a
    /// streaming response. Decode still masks tokens and validates the result.
    pub fn validate(&self) -> Result<()> {
        self.response_format.validate_tool_choice(&self.conversation.tool_choice)?;
        if self.tool_constraints == ToolConstraints::Schema
            && matches!(self.response_format, ResponseFormat::JsonSchema(_))
        {
            return Err(invalid("schema tools and a JSON response format are exclusive"));
        }
        if self.constrained() {
            let options = self.options;
            if options
                .repetition_penalty
                .is_some_and(|value| !value.is_finite() || (value - 1.0).abs() > f32::EPSILON)
                || options.min_tokens.is_some_and(|value| value != 0)
                || options.ignore_eos == Some(true)
            {
                return Err(invalid(
                    "schema mode requires repetition_penalty=1, min_tokens=0 and ignore_eos=false",
                ));
            }
        }
        if let ResponseFormat::JsonSchema(schema) = &self.response_format {
            super::super::constraints::validate_schema(schema)?;
        }
        Ok(())
    }

    fn constrained(&self) -> bool {
        self.tool_constraints == ToolConstraints::Schema
            || matches!(self.response_format, ResponseFormat::JsonSchema(_))
    }

    pub(crate) fn sampling_options(&self) -> models::generation::GenerationOverrides {
        let mut options = self.options;
        if self.constrained() {
            // Grammar constraints take precedence over checkpoint defaults.
            // Explicit incompatible values are rejected by validate().
            options.repetition_penalty.get_or_insert(1.0);
            options.min_tokens.get_or_insert(0);
            options.ignore_eos.get_or_insert(false);
        }
        options
    }
}
