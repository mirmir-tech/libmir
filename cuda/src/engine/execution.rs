use std::time::{Duration, Instant};

use runtime::{
    backend::{
        DecodeOutput, DecodeRequest, LogitsTrace, PrefillOutput, PrefillRequest, SamplingLogits,
        TokenEvent,
    },
    progress::ProgressEvent,
};

use super::{
    CudaEngine,
    model::{DeviceToken, ModelRunner},
};
use crate::{CudaMoeModelSession, Error, Result};

impl CudaEngine {
    pub fn prefill_with_progress(
        &self,
        request: &PrefillRequest,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        if request.prompt_tokens.is_empty() {
            return Err(Error::InvalidDecoderKernel("CUDA prefill prompt is empty"));
        }
        let loaded = self.model(&request.model.id)?;
        let execution_started = Instant::now();
        let mut runner_wait = Duration::ZERO;
        let mut consumed = 0;
        let mut chunks = 0;
        let mut step_table = request.block_table.clone();
        let mut output = None;
        while consumed < request.prompt_tokens.len() {
            let lock_started = Instant::now();
            let mut runner = loaded.prefill_runner()?;
            runner_wait += lock_started.elapsed();
            let remaining = &request.prompt_tokens[consumed..];
            let count = runner.model.prefill_chunk_len(remaining.len());
            step_table.set_token_len(consumed + count);
            runner.model.prefill_chunk(
                request.session_id,
                &remaining[..count],
                consumed,
                &step_table,
            )?;
            consumed += count;
            chunks += 1;
            if consumed == request.prompt_tokens.len() {
                runner.model.finish_prefill(count, request.sampling_logits)?;
                let completed = self.output(&mut runner.model, request.sampling_logits)?;
                runner.selected =
                    completed.token.map(|token| DeviceToken { session: request.session_id, token });
                loaded.register_session(request.session_id)?;
                output = Some(completed);
            }
            drop(runner);
            progress(ProgressEvent::prefill_tokens(consumed, request.prompt_tokens.len()));
            if consumed < request.prompt_tokens.len() {
                std::thread::yield_now();
            }
        }
        let output =
            output.ok_or(Error::InvalidDecoderKernel("CUDA prefill produced no output"))?;
        let execution_elapsed = execution_started.elapsed();
        tracing::debug!(
            backend = "cuda",
            prompt_tokens = request.prompt_tokens.len(),
            runner_wait_ms = runner_wait.as_secs_f64() * 1_000.0,
            execution_ms = execution_elapsed.as_secs_f64() * 1_000.0,
            chunks,
            "completed CUDA prefill request"
        );
        Ok(PrefillOutput {
            accepted_tokens: request.prompt_tokens.len(),
            next_token: output.token,
            trace: Some("cuda.prefill=device-resident-paged-kv".into()),
            logits: output.logits,
            candidates: None,
        })
    }

    pub fn decode_token(&self, request: &DecodeRequest) -> Result<DecodeOutput> {
        let loaded = self.model(&request.model.id)?;
        let mut runner = loaded.decode_runner()?;
        loaded.require_session(request.session_id)?;
        self.decode_with_runner(&mut runner, request)
    }

    pub(super) fn decode_with_runner(
        &self,
        runner: &mut ModelRunner,
        request: &DecodeRequest,
    ) -> Result<DecodeOutput> {
        let selected = DeviceToken {
            session: request.session_id,
            token: request.token_id,
        };
        if runner.selected == Some(selected) {
            runner.model.decode_sampled_for_sampling(
                request.session_id,
                &request.block_table,
                request.sampling_logits,
            )?;
        } else {
            runner.model.decode_for_sampling(
                request.session_id,
                request.token_id,
                &request.block_table,
                request.sampling_logits,
            )?;
        }
        let output = self.output(&mut runner.model, request.sampling_logits)?;
        runner.selected =
            output.token.map(|token| DeviceToken { session: request.session_id, token });
        Ok(DecodeOutput {
            event: TokenEvent {
                token_id: output.token,
                text: "cuda.decode=device-token-pipeline".into(),
                finished: false,
            },
            logits: output.logits,
            candidates: None,
        })
    }

    fn output(&self, session: &mut CudaMoeModelSession, policy: SamplingLogits) -> Result<Output> {
        if device_sampling(policy) {
            let selected = session.sample(policy)?;
            return Ok(Output {
                token: Some(self.backend.read_token(selected)?),
                logits: None,
            });
        }
        let values = self.backend.read_logits(session.logits())?;
        Ok(Output {
            token: None,
            logits: Some(LogitsTrace {
                shape: vec![1, 1, i32::try_from(values.len())?],
                values,
            }),
        })
    }
}

struct Output {
    token: Option<u32>,
    logits: Option<LogitsTrace>,
}

pub(super) const fn device_sampling(policy: SamplingLogits) -> bool {
    matches!(
        policy,
        SamplingLogits::None | SamplingLogits::SampleTopK { .. } | SamplingLogits::Sample { .. }
    )
}
