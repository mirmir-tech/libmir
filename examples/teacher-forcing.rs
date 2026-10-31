//! Replays reference continuations token by token and compares the model's
//! next-token distribution at every position: top-1 agreement, the log
//! probability of the reference token, and the overlap with the reference
//! top-k. A free greedy run then reports where its tokens first diverge.
//!
//! Usage: `teacher-forcing <model> <reference.json> [options]`, where the
//! reference is written by `benchmarks/tools/nemotron/greedy_oracle.py`.
//! Options: `untuned` disables timing-based kernel selection, so repeated
//! runs execute the same kernels; `chunk=N` sets the prefill chunk; and
//! `records=N` replays only the first `N` references.
#![allow(clippy::print_stdout)]

use std::{collections::HashMap, env, fs, path::PathBuf};

use libmir::{GenerationOverrides, Library, RuntimeConfig, SamplingLogits};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    prompt: String,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    top_logprobs: Vec<HashMap<String, f64>>,
}

#[derive(Default)]
struct Totals {
    positions: usize,
    agreements: usize,
    logprob_error: f64,
    top_overlap: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let model = args.next().map(PathBuf::from).ok_or("missing model path")?;
    let reference = args.next().map(PathBuf::from).ok_or("missing reference path")?;
    let mut reference: Reference = serde_json::from_slice(&fs::read(reference)?)?;
    #[allow(unused_mut)]
    let mut config = RuntimeConfig::default();
    for option in args.filter_map(|option| option.into_string().ok()) {
        if let Some(count) = option.strip_prefix("records=") {
            reference.records.truncate(count.parse()?);
        } else if let Some(chunk) = option.strip_prefix("chunk=") {
            #[cfg(feature = "cuda")]
            {
                config.cuda.model_session.prefill_chunk_tokens = chunk.parse()?;
            }
            #[cfg(not(feature = "cuda"))]
            let _ = chunk;
        } else if option == "untuned" {
            #[cfg(feature = "cuda")]
            {
                config.cuda.tuning.mode = runtime::tuning::TuningMode::Disabled;
            }
        } else {
            return Err(format!("unknown option {option}").into());
        }
    }
    let library = Library::new(config);
    let model = library.load(model, GenerationOverrides::default(), &mut |_| {})?;
    let mut totals = Totals::default();
    for record in &reference.records {
        let mut session = model.session();
        let mut logits =
            logits(session.prefill(&record.prompt_ids, SamplingLogits::Full, &mut |_| {})?.logits)?;
        let mut first_miss = None;
        for (step, (&token, top)) in
            record.generated_ids.iter().zip(&record.top_logprobs).enumerate()
        {
            let probabilities = log_softmax(&logits);
            let argmax = argmax(&logits)?;
            totals.positions += 1;
            if argmax == token {
                totals.agreements += 1;
            } else {
                first_miss.get_or_insert(step);
                let reference = |id: u32| top.get(&id.to_string()).copied().unwrap_or(f64::NAN);
                println!(
                    "  miss step={step} reference={token} ours={argmax} \
                     libmir=({:.3}, {:.3}) reference_logprobs=({:.3}, {:.3})",
                    probabilities[token as usize],
                    probabilities[argmax as usize],
                    reference(token),
                    reference(argmax)
                );
            }
            let reference_logprob = top.get(&token.to_string()).copied().unwrap_or(f64::NAN);
            totals.logprob_error += (probabilities[token as usize] - reference_logprob).abs();
            let ours = top_k(&probabilities, top.len());
            totals.top_overlap +=
                count(ours.iter().filter(|id| top.contains_key(&id.to_string())).count())
                    / count(top.len());
            if step + 1 < record.generated_ids.len() {
                logits = self::logits(session.decode(token, SamplingLogits::Full)?.logits)?;
            }
        }
        let divergence = free_divergence(&model, record)?;
        println!(
            "prompt={:?} positions={} first_forced_miss={first_miss:?} free_divergence={divergence:?}",
            record.prompt,
            record.generated_ids.len()
        );
    }
    let positions = count(totals.positions);
    println!(
        "positions={} top1_agreement={:.4} mean_abs_logprob_error={:.4} top_k_overlap={:.4}",
        totals.positions,
        count(totals.agreements) / positions,
        totals.logprob_error / positions,
        totals.top_overlap / positions
    );
    model.unload()?;
    Ok(())
}

/// First position where free greedy decoding leaves the reference tokens.
fn free_divergence(
    model: &libmir::Model,
    record: &Record,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let mut session = model.session();
    let mut logits =
        logits(session.prefill(&record.prompt_ids, SamplingLogits::Full, &mut |_| {})?.logits)?;
    for (step, &expected) in record.generated_ids.iter().enumerate() {
        let token = argmax(&logits)?;
        if token != expected {
            return Ok(Some(step));
        }
        if step + 1 < record.generated_ids.len() {
            logits = self::logits(session.decode(token, SamplingLogits::Full)?.logits)?;
        }
    }
    Ok(None)
}

fn logits(trace: Option<runtime::backend::LogitsTrace>) -> Result<Vec<f32>, &'static str> {
    trace.map(|trace| trace.values).ok_or("backend did not return full logits")
}

fn argmax(values: &[f32]) -> Result<u32, Box<dyn std::error::Error>> {
    let index = values
        .iter()
        .enumerate()
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .ok_or("empty logits")?
        .0;
    Ok(u32::try_from(index)?)
}

fn log_softmax(values: &[f32]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total = values.iter().map(|value| f64::from(value - maximum).exp()).sum::<f64>().ln();
    values.iter().map(|value| f64::from(value - maximum) - total).collect()
}

fn top_k(values: &[f64], k: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.select_nth_unstable_by(k.saturating_sub(1), |left, right| {
        values[*right].total_cmp(&values[*left])
    });
    indices.truncate(k);
    indices
}

fn count(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}
