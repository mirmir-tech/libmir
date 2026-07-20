use std::time::{Duration, Instant};

use foundation::protocol::ChatCompletionRequest;
use models::generation::{
    GenerationChannel, GenerationOverrides, GenerationSettings, GenerationToken, OutputNormalizer,
};
use runtime::{
    backend::{CandidateLogitsTrace, LogitsTrace, SamplingLogits},
    metrics::{GenerationMetrics, GenerationMetricsRecorder},
    sampling::{Sampler, SamplerConfig},
};

use crate::{CancellationToken, Model, ProgressEvent, Result};

#[path = "generation/input.rs"]
mod input;

use input::PreparedGeneration;

#[derive(Debug, Clone)]
/// Completed generation, including separated reasoning, tokens, and timing
/// metrics.
pub struct GenerationOutput {
    /// User-visible assistant text.
    pub text: String,
    /// Reasoning text emitted on the reasoning channel, when present.
    pub reasoning: String,
    /// Generated token identifiers, including a terminal stop token when
    /// emitted.
    pub token_ids: Vec<u32>,
    /// Number of tokens in the prepared prompt.
    pub prompt_tokens: usize,
    /// Stable completion reason such as `"stop"` or `"max_tokens"`.
    pub finish_reason: &'static str,
    /// Prefill, decode, throughput, and cache metrics for this generation.
    pub metrics: GenerationMetrics,
}

impl Model {
    /// Generates a complete response and streams normalized token deltas to
    /// `token`.
    pub fn generate(
        &self,
        request: &ChatCompletionRequest,
        progress: &mut dyn FnMut(ProgressEvent),
        token: &mut dyn FnMut(GenerationToken),
    ) -> Result<GenerationOutput> {
        self.generate_cancellable(request, progress, token, &CancellationToken::default())
    }

    /// Generates like [`Model::generate`], stopping when `cancellation` is
    /// signalled.
    pub fn generate_cancellable(
        &self,
        request: &ChatCompletionRequest,
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
        request: &ChatCompletionRequest,
        encoded_image: &[u8],
        progress: &mut dyn FnMut(ProgressEvent),
        token: &mut dyn FnMut(GenerationToken),
        cancellation: &CancellationToken,
    ) -> Result<GenerationOutput> {
        self.generate_inner(request, Some(encoded_image), progress, token, cancellation)
    }

    fn generate_inner(
        &self,
        request: &ChatCompletionRequest,
        encoded_image: Option<&[u8]>,
        progress: &mut dyn FnMut(ProgressEvent),
        token: &mut dyn FnMut(GenerationToken),
        cancellation: &CancellationToken,
    ) -> Result<GenerationOutput> {
        cancellation.check()?;
        let descriptor = self.descriptor();
        let settings = descriptor.resolve_generation(overrides(request))?;
        let prepared = PreparedGeneration::prepare(self, request, settings, encoded_image)?;
        let prompt_tokens = prepared.token_ids().len();
        let mut metrics = GenerationMetricsRecorder::new();
        metrics.record_prompt(Duration::ZERO, prompt_tokens);
        let tokenizer = descriptor.tokenizer();
        let mut normalizer = OutputNormalizer::new(tokenizer, prepared.prompt_text());
        let vocab_size = tokenizer.vocab_size().min(descriptor.decoder().vocab_size);
        let mut sampler = Sampler::new(sampler_config(settings, request.seed, vocab_size))?;
        let mut session = self.session();
        let sampling = request_sampling(settings, vocab_size, &mut sampler);
        let prefill_started = Instant::now();
        let prefill = prepared.prefill(&mut session, sampling, progress)?;
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
        let mut finish_reason = "max_tokens";
        progress(ProgressEvent::decode_tokens(0, settings.max_tokens));
        while token_ids.len() < settings.max_tokens {
            cancellation.check()?;
            token_ids.push(next);
            progress(ProgressEvent::decode_tokens(token_ids.len(), settings.max_tokens));
            if let Some(history) = history.as_mut() {
                history.push(next);
            }
            let piece = tokenizer.decode(&[next])?;
            if let Some(delta) = normalizer.push(next, piece) {
                match delta.channel {
                    GenerationChannel::Content => text.push_str(&delta.text),
                    GenerationChannel::Reasoning => reasoning.push_str(&delta.text),
                }
                token(delta);
            }
            if tokenizer.stop_token_ids().contains(&next) {
                finish_reason = "stop";
                break;
            }
            if token_ids.len() == settings.max_tokens {
                break;
            }
            let decode_started = Instant::now();
            let output =
                session.decode(next, request_sampling(settings, vocab_size, &mut sampler))?;
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
        Ok(GenerationOutput {
            text,
            reasoning,
            token_ids,
            prompt_tokens,
            finish_reason,
            metrics,
        })
    }
}

fn overrides(request: &ChatCompletionRequest) -> GenerationOverrides {
    GenerationOverrides {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        repetition_penalty: request.repetition_penalty,
    }
}

fn sampler_config(
    settings: GenerationSettings,
    seed: Option<u64>,
    vocab_size: usize,
) -> SamplerConfig {
    SamplerConfig {
        temperature: settings.temperature,
        top_p: settings.top_p,
        top_k: settings.top_k,
        repetition_penalty: settings.repetition_penalty,
        vocab_size: Some(vocab_size),
        seed: seed.unwrap_or_else(|| SamplerConfig::default().seed),
    }
}

fn request_sampling(
    settings: GenerationSettings,
    vocab_size: usize,
    sampler: &mut Sampler,
) -> SamplingLogits {
    let mut sampling = sampling(settings, vocab_size);
    match &mut sampling {
        SamplingLogits::SampleTopK { draw, .. } | SamplingLogits::Sample { draw, .. } => {
            *draw = sampler.draw_unit_f32();
        },
        SamplingLogits::None | SamplingLogits::Full | SamplingLogits::TopK { .. } => {},
    }
    sampling
}

fn sampling(settings: GenerationSettings, vocab_size: usize) -> SamplingLogits {
    let greedy = (settings.temperature <= f32::EPSILON || settings.top_k == 1)
        && settings.repetition_penalty <= 1.0;
    if greedy {
        return SamplingLogits::None;
    }
    if settings.repetition_penalty <= 1.0 && settings.top_k > 0 && settings.top_k < vocab_size {
        if settings.top_p < 1.0 {
            return SamplingLogits::Sample {
                vocab_size,
                temperature: settings.temperature,
                top_p: settings.top_p,
                top_k: settings.top_k,
                draw: 0.0,
            };
        }
        return SamplingLogits::SampleTopK {
            k: settings.top_k,
            vocab_size,
            temperature: settings.temperature,
            draw: 0.0,
        };
    }
    SamplingLogits::Full
}

fn choose(
    token: Option<u32>,
    logits: Option<&LogitsTrace>,
    candidates: Option<&CandidateLogitsTrace>,
    history: &[u32],
    sampler: &mut Sampler,
) -> Result<u32> {
    if let Some(token) = token {
        return Ok(token);
    }
    if let Some(candidates) = candidates {
        return Ok(sampler.sample_candidates_with_history(candidates, history)?);
    }
    let logits = logits.ok_or_else(|| {
        runtime::RuntimeError::Backend("backend returned neither a token nor logits".into())
    })?;
    Ok(sampler.sample_with_history(logits, history)?)
}

#[cfg(test)]
#[path = "generation/tests.rs"]
mod tests;
