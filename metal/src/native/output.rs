use runtime::backend::{CandidateLogitsTrace, LogitsTrace, SamplingLogits};

use super::{
    error::Result,
    model::{LoadedModel, NativeOutput},
};
use crate::engine::{DeviceSampling, sample_u32};
#[derive(Debug)]
pub(super) struct SamplingOutput {
    pub next_token: Option<u32>,
    pub logits: Option<LogitsTrace>,
    pub candidates: Option<CandidateLogitsTrace>,
}

pub(super) fn materialize(
    model: &LoadedModel,
    output: NativeOutput,
    sampling: SamplingLogits,
) -> Result<SamplingOutput> {
    match output {
        NativeOutput::Greedy(next_token) => Ok(SamplingOutput {
            next_token: Some(next_token),
            logits: None,
            candidates: None,
        }),
        NativeOutput::Logits(logits) if sampling == SamplingLogits::None => Ok(SamplingOutput {
            next_token: Some(logits.argmax_u32(model.stream())?),
            logits: None,
            candidates: None,
        }),
        NativeOutput::Logits(logits) if let SamplingLogits::TopK { k, vocab_size } = sampling => {
            top_k(model, &logits, k, vocab_size)
        },
        NativeOutput::Logits(logits)
            if let SamplingLogits::SampleTopK { k, vocab_size, temperature, draw } = sampling =>
        {
            sampled(model, &logits, vocab_size, k, 1.0, temperature, draw)
        },
        NativeOutput::Logits(logits)
            if let SamplingLogits::Sample {
                vocab_size,
                temperature,
                top_p,
                top_k,
                draw,
            } = sampling =>
        {
            sampled(model, &logits, vocab_size, top_k, top_p, temperature, draw)
        },
        NativeOutput::Logits(logits) => {
            let shape = logits.shape()?;
            let values = logits.to_vec_f32(model.stream())?;
            Ok(SamplingOutput {
                next_token: None,
                logits: Some(LogitsTrace { shape, values }),
                candidates: None,
            })
        },
    }
}

fn sampled(
    model: &LoadedModel,
    logits: &crate::engine::Array,
    vocab_size: usize,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    draw: f32,
) -> Result<SamplingOutput> {
    let next_token = sample_u32(
        logits,
        DeviceSampling {
            vocab_size,
            top_k,
            top_p,
            temperature,
            draw,
        },
        model.stream(),
    )?;
    Ok(SamplingOutput {
        next_token: Some(next_token),
        logits: None,
        candidates: None,
    })
}

fn top_k(
    model: &LoadedModel,
    logits: &crate::engine::Array,
    k: usize,
    vocab_size: usize,
) -> Result<SamplingOutput> {
    let candidates = logits.top_k(k, vocab_size, model.stream())?;
    Ok(SamplingOutput {
        next_token: None,
        logits: None,
        candidates: Some(CandidateLogitsTrace {
            token_ids: candidates.token_ids.to_vec_u32(model.stream())?,
            scores: candidates.scores.to_vec_f32(model.stream())?,
        }),
    })
}
