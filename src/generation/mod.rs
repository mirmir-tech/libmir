use std::time::Instant;

use models::generation::{GenerationToken, OutputNormalizer};
use runtime::{metrics::GenerationMetricsRecorder, sampling::Sampler};

use crate::{CancellationToken, Model, ProgressEvent, Result};

mod cycle;
mod input;
mod output;
mod request;
mod sampling;
mod telemetry;

use cycle::CycleRecovery;
use input::PreparedGeneration;
pub use output::GenerationOutput;
use output::{append_delta, finalize_output, finish_metrics, missing_decoder, should_stop};
pub use request::{GenerationRequest, ReasoningCyclePolicy};
use sampling::{choose_timed, request_sampling, sampler_config};
use telemetry::{record_prefill_metrics, record_publish};

impl Model {
    /// Generates and streams a complete normalized response.
    pub fn generate(
        &self,
        request: &GenerationRequest,
        progress: &mut dyn FnMut(ProgressEvent),
        token: &mut dyn FnMut(GenerationToken),
    ) -> Result<GenerationOutput> {
        self.generate_cancellable(request, progress, token, &CancellationToken::default())
    }

    /// Generates like [`Model::generate`], stopping when `cancellation` is
    /// signalled.
    pub fn generate_cancellable(
        &self,
        request: &GenerationRequest,
        progress: &mut dyn FnMut(ProgressEvent),
        token: &mut dyn FnMut(GenerationToken),
        cancellation: &CancellationToken,
    ) -> Result<GenerationOutput> {
        self.generate_inner(request, None, progress, token, cancellation)
    }

    /// Generates from one encoded image using the vision architecture declared
    /// by the loaded checkpoint.
    pub fn generate_image_cancellable(
        &self,
        request: &GenerationRequest,
        encoded_image: &[u8],
        progress: &mut dyn FnMut(ProgressEvent),
        token: &mut dyn FnMut(GenerationToken),
        cancellation: &CancellationToken,
    ) -> Result<GenerationOutput> {
        self.generate_inner(request, Some(encoded_image), progress, token, cancellation)
    }

    fn generate_inner(
        &self,
        request: &GenerationRequest,
        encoded_image: Option<&[u8]>,
        progress: &mut dyn FnMut(ProgressEvent),
        token: &mut dyn FnMut(GenerationToken),
        cancellation: &CancellationToken,
    ) -> Result<GenerationOutput> {
        cancellation.check()?;
        let mut metrics = GenerationMetricsRecorder::new();
        let descriptor = self.descriptor();
        let settings = descriptor.resolve_generation(request.options)?;
        let prompt_started = Instant::now();
        let prepared =
            PreparedGeneration::new(self, &request.conversation, settings, encoded_image)?;
        let prompt_tokens = prepared.token_ids().len();
        metrics.record_prompt(prompt_started.elapsed(), prompt_tokens);
        let prompt_stages = prepared.preparation_timings();
        metrics.record_prompt_stages(prompt_stages.render, prompt_stages.tokenize);
        let output_setup_started = Instant::now();
        let tokenizer = descriptor.tokenizer();
        let stop_token_ids = tokenizer.stop_token_ids();
        let mut normalizer = OutputNormalizer::new(tokenizer, prepared.prompt_text());
        let mut text_decoder = tokenizer.decoder();
        let decoder = descriptor.decoder().ok_or_else(missing_decoder)?;
        let vocab_size = tokenizer.vocab_size().min(decoder.vocab_size);
        let output_setup = output_setup_started.elapsed();
        let sampler_started = Instant::now();
        let mut sampler = Sampler::new(sampler_config(settings, request.seed, vocab_size))?;
        let sampler_setup = sampler_started.elapsed();
        let session_started = Instant::now();
        let mut session = self.session();
        let sampling = request_sampling(settings, vocab_size, &mut sampler);
        let harmony_exit = harmony_exit(request, descriptor)?;
        let mut cycle_recovery =
            CycleRecovery::new(settings, request.seed, vocab_size, sampling, harmony_exit)?;
        metrics.record_setup_stages(output_setup, sampler_setup, session_started.elapsed());
        let prefill_started = Instant::now();
        let prefill = prepared.prefill(&mut session, settings.max_tokens, sampling, progress)?;
        cancellation.check()?;
        record_prefill_metrics(&mut metrics, prefill_started, prompt_tokens, &prefill);
        let first_started = Instant::now();
        let mut published = false;
        let mut history = sampling.requires_history().then(|| prepared.token_ids().to_vec());
        let mut next = choose_timed(
            &mut metrics,
            prefill.next_token,
            prefill.logits.as_ref(),
            prefill.candidates.as_ref(),
            history.as_deref().unwrap_or_default(),
            &mut sampler,
        )?;
        let mut token_ids = Vec::with_capacity(settings.max_tokens);
        let (mut text, mut reasoning, mut tool_calls) =
            (String::new(), String::new(), String::new());
        let mut finish_reason = "max_tokens";
        while token_ids.len() < settings.max_tokens {
            cancellation.check()?;
            token_ids.push(next);
            if let Some(history) = history.as_mut() {
                history.push(next);
            }
            let piece = text_decoder.step(next)?.unwrap_or_default();
            let delta = normalizer.push(next, piece);
            if let Some(delta) = delta.as_ref() {
                append_delta(delta, &mut text, &mut reasoning, &mut tool_calls);
            }
            let stopped = should_stop(settings, token_ids.len(), next, &stop_token_ids);
            if stopped {
                finish_reason = "stop";
            }
            let finished = stopped || token_ids.len() == settings.max_tokens;
            let pending = if finished {
                None
            } else {
                cycle_recovery.observe(prepared.token_ids(), &token_ids);
                let started = Instant::now();
                let sampling = request_sampling(settings, vocab_size, &mut sampler);
                let sampling = cycle_recovery.sampling(sampling);
                Some((started, session.start_decode(next, sampling)?))
            };
            if token_ids.len() == 1 {
                progress(ProgressEvent::decode_tokens(0, settings.max_tokens));
            }
            progress(ProgressEvent::decode_tokens(token_ids.len(), settings.max_tokens));
            if let Some(delta) = delta {
                token(delta);
                record_publish(&mut metrics, first_started, token_ids.len(), &mut published);
            }
            if finished {
                break;
            }
            let Some((decode_started, pending)) = pending else {
                break;
            };
            let output = session.finish_decode(pending)?;
            metrics.record_decode(decode_started.elapsed());
            next = cycle_recovery.choose(
                &mut metrics,
                &output,
                history.as_deref().unwrap_or_default(),
                &mut sampler,
            )?;
        }
        let metrics = finish_metrics(&mut metrics, token_ids.len(), &session);
        Ok(finalize_output(
            text, reasoning, tool_calls, token_ids, prompt_tokens, finish_reason, metrics,
        ))
    }
}

fn harmony_exit(
    request: &GenerationRequest,
    descriptor: &crate::ModelDescriptor,
) -> Result<Option<(usize, Vec<u32>)>> {
    let ReasoningCyclePolicy::ExitReasoning { min_tokens } = request.reasoning_cycle else {
        return Ok(None);
    };
    if descriptor.metadata().model_type.as_deref() != Some("gpt_oss") {
        return Ok(None);
    }
    Ok(descriptor
        .tokenizer()
        .harmony_reasoning_exit_tokens()?
        .map(|tokens| (min_tokens, tokens)))
}

#[cfg(test)]
mod tests;
