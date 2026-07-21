use std::time::{Duration, Instant};

use runtime::{
    backend::{DecodeOutput, DecodeRequest, PrefillOutput, PrefillRequest},
    progress::ProgressEvent,
};

use super::{
    CudaEngine,
    model::{DeviceToken, ModelExecution, ModelRunner},
};
use crate::{Error, Result};

mod output;

use output::decode_output;
pub(super) use output::{Output, device_sampling, generation_output};

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
            let (count, completed) = {
                let ModelExecution::Generation(generation) = &mut runner.execution else {
                    return Err(Error::State("CUDA task is not a generation runner".into()));
                };
                let count = generation.prefill_chunk_len(remaining.len());
                if count == 0 || count > remaining.len() {
                    return Err(Error::InvalidDecoderKernel(
                        "CUDA lowering returned an invalid prefill chunk",
                    ));
                }
                step_table.set_token_len(consumed + count);
                let final_chunk = consumed + count == request.prompt_tokens.len();
                let completed = generation.prefill_chunk(
                    &self.backend,
                    request,
                    &remaining[..count],
                    consumed,
                    &step_table,
                    final_chunk,
                )?;
                (count, completed)
            };
            consumed += count;
            chunks += 1;
            if let Some(completed) = completed {
                runner.selected =
                    completed.token.map(|token| DeviceToken { session: request.session_id, token });
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
        loaded.register_session(request.session_id)?;
        tracing::debug!(
            backend = "cuda",
            prompt_tokens = request.prompt_tokens.len(),
            runner_wait_ms = runner_wait.as_secs_f64() * 1_000.0,
            execution_ms = execution_started.elapsed().as_secs_f64() * 1_000.0,
            chunks,
            "completed CUDA prefill request"
        );
        Ok(PrefillOutput {
            accepted_tokens: request.prompt_tokens.len(),
            next_token: output.token,
            trace: Some("cuda.prefill=semantic-operation-plan".into()),
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
        let use_device_token = runner.selected == Some(selected);
        let ModelExecution::Generation(generation) = &mut runner.execution else {
            return Err(Error::State("CUDA task is not a generation runner".into()));
        };
        let output = generation.decode(&self.backend, request, use_device_token)?;
        runner.selected =
            output.token.map(|token| DeviceToken { session: request.session_id, token });
        Ok(decode_output(output))
    }
}
