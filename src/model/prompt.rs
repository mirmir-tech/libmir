use std::time::Instant;

use foundation::conversation::Conversation;
use models::{execution::ModelTask, generation::GenerationSettings};

use super::{
    ModelDescriptor, PreparedPrompt, PromptPreparationTimings, task_mismatch, validate_context,
};
use crate::{Error, Result};

impl ModelDescriptor {
    /// Renders and tokenizes a request, validating it against the model context
    /// window.
    pub fn prepare(&self, conversation: &Conversation) -> Result<PreparedPrompt> {
        if !matches!(self.task_plan.task(), ModelTask::Generation) {
            return Err(task_mismatch("generation", &self.task_plan));
        }
        self.prepare_with_settings(conversation, self.generation)
    }

    pub(crate) fn prepare_with_settings(
        &self,
        conversation: &Conversation,
        generation: GenerationSettings,
    ) -> Result<PreparedPrompt> {
        self.prepare_with_reasoning(conversation, generation, crate::ReasoningMode::ModelDefault)
    }

    pub(crate) fn prepare_with_reasoning(
        &self,
        conversation: &Conversation,
        generation: GenerationSettings,
        reasoning: crate::ReasoningMode,
    ) -> Result<PreparedPrompt> {
        let render_started = Instant::now();
        let prompt = self.template.render_with_reasoning(conversation, reasoning)?;
        let render = render_started.elapsed();
        let tokenize_started = Instant::now();
        let tokens = self
            .tokenizer
            .encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?;
        let tokenize = tokenize_started.elapsed();
        if tokens.token_ids.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        validate_context(tokens.token_ids.len(), generation.max_tokens, self.metadata.context_len)?;
        let cache_checkpoints =
            self.cache_checkpoints(conversation, &tokens.token_ids, reasoning)?;
        Ok(PreparedPrompt {
            prompt,
            tokens,
            cache_checkpoints,
            timings: PromptPreparationTimings { render, tokenize },
        })
    }
}
