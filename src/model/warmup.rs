use models::execution::ModelTask;
use runtime::backend::SamplingLogits;

use super::Model;
use crate::{CancellationToken, Error, ProgressEvent, Result, RuntimeError};

const PROFILE_CONTEXT_TOKENS: usize = 2_048;
const PROFILE_DECODE_STEPS: usize = 2;
const PROFILE_SESSIONS: usize = 2;

impl Model {
    /// Warms reusable accelerator execution profiles before serving requests.
    ///
    /// The workload is derived only from the model context and tokenizer. Two
    /// sessions exercise both fresh and reusable-prefix execution,
    /// including the first two decode context buckets above the prompt length.
    /// Backends may additionally warm exact full and interleaved prefill shapes
    /// before startup tuning is sealed.
    pub fn warm_execution_profiles(&self, progress: &mut dyn FnMut(ProgressEvent)) -> Result<()> {
        let result = self.warm_execution_profiles_inner(progress).and_then(|()| {
            // The widest decode batch allocates on first use; build it before
            // startup tuning seals and memory is measured.
            let rows = self.inner.config.scheduler.max_batch_requests.max(1);
            Ok(self.engine().warm_concurrency(self.handle(), rows)?)
        });
        let finish = self.engine().finish_startup_tuning(self.handle());
        // Startup tuning is sealed and its scratch is resident: measure now,
        // even after a failed warm-up, so the model never serves on the
        // provisional cache.
        let finalized = self.finalize_kv_cache();
        result.and(finish)?;
        finalized
    }

    fn warm_execution_profiles_inner(&self, progress: &mut dyn FnMut(ProgressEvent)) -> Result<()> {
        if !matches!(self.descriptor().task(), ModelTask::Generation) {
            return Ok(());
        }
        let Some(tokens) = profile_context(self.descriptor().metadata().context_len) else {
            return Ok(());
        };
        let seed = self
            .descriptor()
            .tokenizer()
            .encode_with_special_tokens("Warm accelerator execution profiles.", false)?
            .token_ids;
        seed.first().copied().ok_or(Error::EmptyPrompt)?;
        let prompt = seed.into_iter().cycle().take(tokens).collect::<Vec<_>>();
        tracing::info!(
            model = %self.handle().id,
            prompt_tokens = prompt.len(),
            sessions = PROFILE_SESSIONS,
            decode_steps = PROFILE_DECODE_STEPS,
            "warming accelerator execution profiles"
        );
        let profiles = self
            .engine()
            .prefill_profile_shapes(self.handle(), self.inner.config.scheduler.max_batch_tokens)?
            .into_iter()
            .filter_map(|shape| {
                profile_checkpoint(
                    shape,
                    self.inner.config.kv_cache.block_size,
                    self.descriptor().metadata().context_len,
                )
            })
            .collect::<Vec<_>>();
        let base_total = PROFILE_SESSIONS * (PROFILE_DECODE_STEPS + 1);
        let total = base_total + profiles.len();
        progress(ProgressEvent::warmup(0, total, "warming accelerator execution profiles"));
        for session_index in 0..PROFILE_SESSIONS {
            let prompt = if session_index == 0 {
                prompt.as_slice()
            } else {
                &prompt[..prompt.len().saturating_sub(1)]
            };
            let mut session = self.session();
            let output = session.prefill(prompt, SamplingLogits::None, &mut |_| {})?;
            let mut token = required_token(output.next_token)?;
            let completed = session_index * (PROFILE_DECODE_STEPS + 1) + 1;
            progress(ProgressEvent::warmup(completed, total, "prefill profile is warm"));
            for step in 0..PROFILE_DECODE_STEPS {
                token =
                    required_token(session.decode(token, SamplingLogits::None)?.event.token_id)?;
                progress(ProgressEvent::warmup(
                    completed + step + 1,
                    total,
                    "decode profile is warm",
                ));
            }
        }
        for (index, (shape, tokens)) in profiles.into_iter().enumerate() {
            // Each shape needs a distinct prefix: otherwise the preceding
            // profile can satisfy it from cache without executing
            // the intended projection. A checkpoint preserves the
            // exact shape with other KV block sizes.
            let seed = self
                .descriptor()
                .tokenizer()
                .encode_with_special_tokens(
                    &format!("Calibrate prefill {shape} accelerator execution."),
                    false,
                )?
                .token_ids;
            seed.first().copied().ok_or(Error::EmptyPrompt)?;
            let prompt = seed.into_iter().cycle().take(tokens).collect::<Vec<_>>();
            self.session().prefill_generation_reserved(
                &prompt,
                &[shape],
                0,
                SamplingLogits::None,
                &CancellationToken::default(),
                &mut |_| {},
            )?;
            tracing::info!(model = %self.handle().id, shape, "exact prefill profile is warm");
            progress(ProgressEvent::warmup(
                base_total + index + 1,
                total,
                "exact prefill profile is warm",
            ));
        }
        tracing::info!(model = %self.handle().id, "accelerator execution profiles are warm");
        Ok(())
    }
}

fn profile_checkpoint(shape: usize, block_tokens: usize, context: usize) -> Option<(usize, usize)> {
    let tokens = shape.checked_add(block_tokens.max(1))?;
    (shape > 0 && tokens <= context.checked_sub(PROFILE_DECODE_STEPS)?).then_some((shape, tokens))
}

fn required_token(token: Option<u32>) -> Result<u32> {
    token.ok_or_else(|| {
        RuntimeError::Backend("profile warmup produced no device token".into()).into()
    })
}

fn profile_context(context: usize) -> Option<usize> {
    let available = context.checked_sub(PROFILE_DECODE_STEPS)?;
    (available > 0).then_some(available.min(PROFILE_CONTEXT_TOKENS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_interleaved_checkpoint_and_reserves_a_tail() {
        assert_eq!(profile_checkpoint(960, 16, 40960), Some((960, 976)));
        assert_eq!(profile_checkpoint(960, 32, 994), Some((960, 992)));
        assert_eq!(profile_checkpoint(960, 32, 993), None);
        assert_eq!(profile_checkpoint(0, 16, 40960), None);
        assert_eq!(profile_checkpoint(960, 16, 1), None);
        assert_eq!(profile_checkpoint(usize::MAX, 16, usize::MAX), None);
    }

    #[test]
    fn bounds_profile_context_and_reserves_decode_positions() {
        assert_eq!(profile_context(40_960), Some(2_048));
        assert_eq!(profile_context(1_024), Some(1_022));
        assert_eq!(profile_context(2), None);
    }
}
