use models::execution::ModelTask;
use runtime::backend::SamplingLogits;

use super::Model;
use crate::{Error, ProgressEvent, Result, runtime::RuntimeError};

const PROFILE_CONTEXT_TOKENS: usize = 2_048;
const PROFILE_DECODE_STEPS: usize = 2;
const PROFILE_SESSIONS: usize = 2;

impl Model {
    /// Warms reusable accelerator execution profiles before serving requests.
    ///
    /// The workload is derived only from the model context and tokenizer. Two
    /// identical sessions exercise both fresh and reusable-prefix execution,
    /// including the first two decode context buckets above the prompt length.
    pub fn warm_execution_profiles(&self, progress: &mut dyn FnMut(ProgressEvent)) -> Result<()> {
        let result = self.warm_execution_profiles_inner(progress);
        let finish = self.engine().finish_startup_tuning(self.handle());
        result.and(finish)
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
        for session_index in 0..PROFILE_SESSIONS {
            let prompt = if session_index == 0 {
                prompt.as_slice()
            } else {
                &prompt[..prompt.len().saturating_sub(1)]
            };
            let mut session = self.session();
            let output = session.prefill(prompt, SamplingLogits::None, progress)?;
            let mut token = required_token(output.next_token)?;
            for step in 0..PROFILE_DECODE_STEPS {
                token =
                    required_token(session.decode(token, SamplingLogits::None)?.event.token_id)?;
                progress(ProgressEvent::decode_tokens(
                    session_index * PROFILE_DECODE_STEPS + step + 1,
                    PROFILE_SESSIONS * PROFILE_DECODE_STEPS,
                ));
            }
        }
        tracing::info!(model = %self.handle().id, "accelerator execution profiles are warm");
        Ok(())
    }
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
    fn bounds_profile_context_and_reserves_decode_positions() {
        assert_eq!(profile_context(40_960), Some(2_048));
        assert_eq!(profile_context(1_024), Some(1_022));
        assert_eq!(profile_context(2), None);
    }
}
