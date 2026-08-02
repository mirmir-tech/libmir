use models::generation::GenerationSettings;
use runtime::{
    backend::{CandidateLogitsTrace, LogitsTrace, SamplingLogits},
    sampling::{Sampler, SamplerConfig},
};

use crate::Result;

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

pub(super) fn sampling(settings: GenerationSettings, vocab_size: usize) -> SamplingLogits {
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
