use std::time::{Duration, Instant};

use models::generation::{
    GenerationChannel, GenerationSettings, GenerationToken, OutputNormalizer,
};
use runtime::{metrics::GenerationMetricsRecorder, sampling::Sampler};

use crate::{CancellationToken, Error, Model, ProgressEvent, Result};

mod input;
mod output;
mod request;
mod sampling;

use input::PreparedGeneration;
pub use output::GenerationOutput;
pub use request::GenerationRequest;
use sampling::{choose, request_sampling, sampler_config};

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
        let descriptor = self.descriptor();
        let settings = descriptor.resolve_generation(request.options)?;
        let prepared =
            PreparedGeneration::new(self, &request.conversation, settings, encoded_image)?;
        let prompt_tokens = prepared.token_ids().len();
        let mut metrics = GenerationMetricsRecorder::new();
        metrics.record_prompt(Duration::ZERO, prompt_tokens);
        let tokenizer = descriptor.tokenizer();
        let stop_token_ids = tokenizer.stop_token_ids();
        let mut normalizer = OutputNormalizer::new(tokenizer, prepared.prompt_text());
        let mut text_decoder = tokenizer.decoder();
        let decoder = descriptor.decoder().ok_or_else(missing_decoder)?;
        let vocab_size = tokenizer.vocab_size().min(decoder.vocab_size);
        let mut sampler = Sampler::new(sampler_config(settings, request.seed, vocab_size))?;
        let mut session = self.session();
        let sampling = request_sampling(settings, vocab_size, &mut sampler);
        let prefill_started = Instant::now();
        let prefill = prepared.prefill(&mut session, settings.max_tokens, sampling, progress)?;
        cancellation.check()?;
        metrics.record_prefill(prefill_started.elapsed(), prompt_tokens);
        let mut history = sampling.requires_history().then(|| prepared.token_ids().to_vec());
        let mut next = choose(
            prefill.next_token,
            prefill.logits.as_ref(),
            prefill.candidates.as_ref(),
            history.as_deref().unwrap_or_default(),
            &mut sampler,
        )?;
        let mut token_ids = Vec::with_capacity(settings.max_tokens);
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = String::new();
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
                match delta.channel {
                    GenerationChannel::Content => text.push_str(&delta.text),
                    GenerationChannel::Reasoning => reasoning.push_str(&delta.text),
                    GenerationChannel::ToolCalls => tool_calls.push_str(&delta.text),
                }
            }
            let stopped = should_stop(settings, token_ids.len(), next, &stop_token_ids);
            if stopped {
                finish_reason = "stop";
            }
            let finished = stopped || token_ids.len() == settings.max_tokens;
            let pending = if finished {
                None
            } else {
                let started = Instant::now();
                let sampling = request_sampling(settings, vocab_size, &mut sampler);
                Some((started, session.start_decode(next, sampling)?))
            };
            if token_ids.len() == 1 {
                progress(ProgressEvent::decode_tokens(0, settings.max_tokens));
            }
            progress(ProgressEvent::decode_tokens(token_ids.len(), settings.max_tokens));
            if let Some(delta) = delta {
                token(delta);
            }
            if finished {
                break;
            }
            let Some((decode_started, pending)) = pending else {
                break;
            };
            let output = session.finish_decode(pending)?;
            metrics.record_decode(decode_started.elapsed());
            next = choose(
                output.event.token_id,
                output.logits.as_ref(),
                output.candidates.as_ref(),
                history.as_deref().unwrap_or_default(),
                &mut sampler,
            )?;
        }
        metrics.record_generated(token_ids.len());
        let metrics = metrics.snapshot(session.cache_stats());
        if !tool_calls.is_empty() {
            finish_reason = "tool_calls";
        }
        Ok(GenerationOutput {
            text,
            reasoning,
            tool_calls,
            token_ids,
            prompt_tokens,
            finish_reason,
            metrics,
        })
    }
}

fn missing_decoder() -> Error {
    Error::TaskMismatch {
        requested: "generation",
        actual: "sequence scoring",
    }
}

fn should_stop(
    settings: GenerationSettings,
    generated_tokens: usize,
    token: u32,
    stop_tokens: &[u32],
) -> bool {
    !settings.ignore_eos && generated_tokens >= settings.min_tokens && stop_tokens.contains(&token)
}

#[cfg(test)]
mod tests;
