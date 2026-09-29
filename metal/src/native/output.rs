use runtime::backend::{CandidateLogitsTrace, LogitsTrace, SamplingLogits, TokenMask};

use super::{
    error::Result,
    model::{LoadedModel, NativeOutput},
};
use crate::engine::{DeviceSampling, MaskedSampling, TokenAllowlist, sample_u32};
#[derive(Debug)]
pub(super) struct SamplingOutput {
    pub next_token: Option<u32>,
    pub logits: Option<LogitsTrace>,
    pub candidates: Option<CandidateLogitsTrace>,
}

pub(super) fn materialize(
    model: &LoadedModel,
    output: NativeOutput,
    sampling: &SamplingLogits,
) -> Result<SamplingOutput> {
    let logits = match (output, sampling) {
        (NativeOutput::Greedy(_), SamplingLogits::Masked { .. }) => {
            return Err(super::error::Error::InvalidDecodeBatch(
                "schema masking requires full device logits".into(),
            ));
        },
        (NativeOutput::Greedy(next_token), _) => {
            return Ok(SamplingOutput {
                next_token: Some(next_token),
                logits: None,
                candidates: None,
            });
        },
        (NativeOutput::Logits(logits), _) => logits,
    };
    match *sampling {
        SamplingLogits::Masked { ref mask, sampling } => masked(model, &logits, mask, sampling),
        SamplingLogits::None => Ok(SamplingOutput {
            next_token: Some(logits.argmax_u32(model.stream())?),
            logits: None,
            candidates: None,
        }),
        SamplingLogits::TopK { k, vocab_size } => top_k(model, &logits, k, vocab_size),
        SamplingLogits::SampleTopK { k, vocab_size, temperature, draw } => {
            sampled(model, &logits, vocab_size, k, 1.0, temperature, draw)
        },
        SamplingLogits::Sample {
            vocab_size,
            temperature,
            top_p,
            top_k,
            draw,
        } if super::step::supports_device_token(sampling) => {
            sampled(model, &logits, vocab_size, top_k, top_p, temperature, draw)
        },
        SamplingLogits::Sample { .. } | SamplingLogits::Full => {
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

/// The grammar needs each token on the host before it can mask the next step,
/// so masked rows read one scalar instead of the full logits.
fn masked(
    model: &LoadedModel,
    logits: &crate::engine::Array,
    mask: &TokenMask,
    sampling: runtime::backend::DeviceSampling,
) -> Result<SamplingOutput> {
    let sampling = match sampling {
        runtime::backend::DeviceSampling::Greedy => MaskedSampling::Greedy,
        runtime::backend::DeviceSampling::Random { temperature, top_p, top_k, draw } => {
            MaskedSampling::Random { top_k, top_p, temperature, draw }
        },
    };
    let allowed = TokenAllowlist { words: mask.words(), vocab: mask.vocab() };
    Ok(SamplingOutput {
        next_token: Some(logits.masked_token_u32(allowed, sampling, model.stream())?),
        logits: None,
        candidates: None,
    })
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
