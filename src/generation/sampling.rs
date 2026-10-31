use std::time::Instant;

use models::generation::GenerationSettings;
use runtime::{
    backend::{CandidateLogitsTrace, LogitsTrace, PrefillOutput, SamplingLogits},
    metrics::GenerationMetricsRecorder,
    sampling::{Sampler, SamplerConfig},
};

use crate::{Model, Result};

pub(super) fn sampler_config(
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

pub(super) fn request_sampling(
    model: &Model,
    settings: GenerationSettings,
    vocab_size: usize,
    sampler: &mut Sampler,
) -> SamplingLogits {
    let device_nucleus = model.engine().samples_unbounded_nucleus();
    let mut sampling = sampling(settings, vocab_size, device_nucleus);
    match &mut sampling {
        SamplingLogits::SampleTopK { draw, .. } | SamplingLogits::Sample { draw, .. } => {
            *draw = sampler.draw_unit_f32();
        },
        SamplingLogits::Masked { .. }
        | SamplingLogits::None
        | SamplingLogits::Full
        | SamplingLogits::TopK { .. } => {},
    }
    sampling
}

/// `device_nucleus` lets top-p without a top-k bound sample on the device;
/// otherwise it reads full logits.
pub(super) fn sampling(
    settings: GenerationSettings,
    vocab_size: usize,
    device_nucleus: bool,
) -> SamplingLogits {
    let greedy = (settings.temperature <= f32::EPSILON || settings.top_k == 1)
        && settings.repetition_penalty <= 1.0;
    if greedy {
        return SamplingLogits::None;
    }
    if settings.repetition_penalty <= 1.0 {
        if settings.top_k == 0 && (settings.top_p >= 1.0 || device_nucleus) {
            return SamplingLogits::Sample {
                vocab_size,
                temperature: settings.temperature,
                top_p: settings.top_p,
                top_k: 0,
                draw: 0.0,
            };
        }
        if settings.top_k > 0 && settings.top_k < vocab_size && settings.top_p < 1.0 {
            return SamplingLogits::Sample {
                vocab_size,
                temperature: settings.temperature,
                top_p: settings.top_p,
                top_k: settings.top_k,
                draw: 0.0,
            };
        }
        if settings.top_k > 0 && settings.top_k < vocab_size {
            return SamplingLogits::SampleTopK {
                k: settings.top_k,
                vocab_size,
                temperature: settings.temperature,
                draw: 0.0,
            };
        }
    }
    SamplingLogits::Full
}

pub(super) fn choose(
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

pub(super) fn choose_timed(
    metrics: &mut GenerationMetricsRecorder,
    token: Option<u32>,
    logits: Option<&LogitsTrace>,
    candidates: Option<&CandidateLogitsTrace>,
    history: &[u32],
    sampler: &mut Sampler,
) -> Result<u32> {
    let started = Instant::now();
    let result = choose(token, logits, candidates, history, sampler);
    metrics.record_sampling(started.elapsed());
    result
}

pub(super) fn choose_prefill(
    prefill: &PrefillOutput,
    history: Option<&[u32]>,
    sampler: &mut Sampler,
    metrics: &mut GenerationMetricsRecorder,
) -> Result<u32> {
    choose_timed(
        metrics,
        prefill.next_token,
        prefill.logits.as_ref(),
        prefill.candidates.as_ref(),
        history.unwrap_or_default(),
        sampler,
    )
}
