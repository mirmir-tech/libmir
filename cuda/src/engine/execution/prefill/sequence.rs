use std::time::{Duration, Instant};

use runtime::backend::{PrefillOutput, PrefillRequest};

use super::{Sequence, prefix::PrefixReuse};
use crate::{Error, Result};

impl Sequence {
    pub(super) fn new(request: PrefillRequest, prefix: PrefixReuse, runner_wait: Duration) -> Self {
        let step_table = request.block_table.clone();
        Self {
            request,
            consumed: prefix.tokens,
            chunks: 0,
            prefix_tokens: prefix.tokens,
            checkpoint_restored: prefix.checkpoint_restored,
            runner_wait,
            completed_at: None,
            step_table,
            output: None,
        }
    }

    pub(super) fn pending(&self) -> bool {
        self.consumed < self.request.prompt_tokens.len()
    }

    pub(super) fn checkpoint_distance(&self) -> usize {
        let declared = self
            .request
            .cache_checkpoints
            .iter()
            .copied()
            .find(|checkpoint| *checkpoint > self.consumed)
            .map(|checkpoint| checkpoint - self.consumed);
        let terminal = self
            .request
            .terminal_cache_checkpoint()
            .filter(|checkpoint| *checkpoint > self.consumed)
            .map(|checkpoint| checkpoint - self.consumed);
        declared.into_iter().chain(terminal).min().unwrap_or(usize::MAX)
    }

    pub(super) fn finish(
        mut self,
        loaded: &crate::engine::model::LoadedModel,
        started: Instant,
    ) -> Result<PrefillOutput> {
        let elapsed = started.elapsed();
        let output = self
            .output
            .take()
            .ok_or(Error::InvalidDecoderKernel("CUDA prefill produced no output"))?;
        let completion = self
            .completed_at
            .map_or(elapsed, |completed| completed.saturating_duration_since(started));
        loaded.register_session(self.request.session_id)?;
        tracing::debug!(
            backend = "cuda",
            session = %self.request.session_id,
            prompt_tokens = self.request.prompt_tokens.len(),
            prefix_cache_tokens = self.prefix_tokens,
            checkpoint_restored = self.checkpoint_restored,
            runner_wait_ms = self.runner_wait.as_secs_f64() * 1_000.0,
            completion_ms = completion.as_secs_f64() * 1_000.0,
            cohort_elapsed_ms = elapsed.as_secs_f64() * 1_000.0,
            chunks = self.chunks,
            "completed CUDA prefill request"
        );
        Ok(PrefillOutput {
            accepted_tokens: self.request.prompt_tokens.len(),
            next_token: output.token,
            trace: Some(format!(
                "cuda.prefill=token-budget-{};prefix_cache_tokens={};checkpoint_restored={}",
                if self.checkpoint_restored {
                    "checkpoint-completion-first"
                } else {
                    "round-robin"
                },
                self.prefix_tokens,
                self.checkpoint_restored
            )),
            logits: output.logits,
            candidates: None,
        })
    }
}
