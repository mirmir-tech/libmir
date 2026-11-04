//! Greedy decoding of several concurrent sessions against the same prompts
//! decoded alone. The concurrent sessions step together, so the scheduler
//! runs their decode as one packed batch; their tokens should match the solo
//! runs except where the solo logits are nearly tied. Reports, per row, the
//! first diverging step with both runs' top-two logit margins there, and the
//! wall time of solo and batched decode.
//!
//! Usage: `batch-parity <model> <reference.json> [rows] [steps] [options]`,
//! with prompts from `benchmarks/tools/nemotron/greedy_oracle.py`. Options:
//! `untuned` disables timing-based kernel selection; `batched-first` runs the
//! concurrent sessions first, so their packed prefill computes the prompts
//! instead of reusing the prefix cache the solo runs published.
#![allow(clippy::print_stdout)]

use std::{
    env, fs,
    path::PathBuf,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use libmir::{GenerationOverrides, Library, Model, RuntimeConfig, SamplingLogits};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    prompt_ids: Vec<u32>,
}

/// Greedy tokens of one sequence and the top-1 minus top-2 logit at each.
struct Trace {
    tokens: Vec<u32>,
    margins: Vec<f32>,
    decode: Duration,
}

type Failure = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Failure> {
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init(),
    );
    let mut args = env::args().skip(1);
    let model = args.next().map(PathBuf::from).ok_or("missing model path")?;
    let reference = args.next().ok_or("missing reference path")?;
    let rows = args.next().map_or(Ok(3), |value| value.parse())?;
    let steps = args.next().map_or(Ok(64), |value| value.parse())?;
    let options = args.collect::<Vec<_>>();
    if let Some(option) = options
        .iter()
        .find(|option| !matches!(option.as_str(), "untuned" | "batched-first"))
    {
        return Err(format!("unknown option {option}").into());
    }
    let untuned = options.iter().any(|option| option == "untuned");
    let batched_first = options.iter().any(|option| option == "batched-first");
    let reference: Reference = serde_json::from_slice(&fs::read(reference)?)?;
    let prompts = reference
        .records
        .into_iter()
        .take(rows)
        .map(|r| r.prompt_ids)
        .collect::<Vec<_>>();
    if prompts.len() < rows {
        return Err("the reference has fewer prompts than rows".into());
    }
    #[allow(unused_mut)]
    let mut config = RuntimeConfig::default();
    config.scheduler.max_batch_requests = rows;
    #[cfg(feature = "cuda")]
    if untuned {
        config.cuda.tuning.mode = runtime::tuning::TuningMode::Disabled;
    }
    #[cfg(not(feature = "cuda"))]
    let _ = untuned;
    let model = Library::new(config).load(model, GenerationOverrides::default(), &mut |_| {})?;
    let (solo, batched) = if batched_first {
        let batched = concurrent(&model, &prompts, steps)?;
        (sequential(&model, &prompts, steps)?, batched)
    } else {
        let solo = sequential(&model, &prompts, steps)?;
        (solo, concurrent(&model, &prompts, steps)?)
    };
    for (row, (solo, batched)) in solo.iter().zip(&batched).enumerate() {
        let divergence = solo.tokens.iter().zip(&batched.tokens).position(|(a, b)| a != b);
        let margins = divergence.map(|step| (solo.margins[step], batched.margins[step]));
        println!(
            "row={row} prompt_tokens={} steps={steps} divergence={divergence:?} margins={margins:?}",
            prompts[row].len()
        );
    }
    let solo_ms = solo.iter().map(|trace| trace.decode.as_secs_f64()).sum::<f64>() * 1e3;
    let batched_ms = batched.iter().map(|t| t.decode.as_secs_f64()).fold(0.0, f64::max) * 1e3;
    let tokens = f64::from(u32::try_from(rows * steps)?);
    println!(
        "solo_decode_ms={solo_ms:.1} solo_tokens_per_second={:.2} batched_decode_ms={batched_ms:.1} \
         batched_tokens_per_second={:.2}",
        tokens / solo_ms * 1e3,
        tokens / batched_ms * 1e3
    );
    model.unload()?;
    Ok(())
}

/// Each prompt decoded alone, one after another.
fn sequential(model: &Model, prompts: &[Vec<u32>], steps: usize) -> Result<Vec<Trace>, Failure> {
    prompts.iter().map(|prompt| greedy(model, prompt, steps, None)).collect()
}

/// All prompts decoded at once, one thread each, stepping together.
fn concurrent(model: &Model, prompts: &[Vec<u32>], steps: usize) -> Result<Vec<Trace>, Failure> {
    let barrier = Arc::new(Barrier::new(prompts.len()));
    prompts
        .iter()
        .map(|prompt| {
            let (model, prompt, barrier) = (model.clone(), prompt.clone(), Arc::clone(&barrier));
            thread::spawn(move || greedy(&model, &prompt, steps, Some(&barrier)))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|handle| handle.join().map_err(|_| "decode thread panicked")?)
        .collect()
}

/// Prefills `prompt` and decodes `steps` greedy tokens. With a barrier, every
/// decode step waits for the other sequences, so they arrive together.
fn greedy(
    model: &Model,
    prompt: &[u32],
    steps: usize,
    barrier: Option<&Barrier>,
) -> Result<Trace, Failure> {
    let mut session = model.session();
    let mut logits = full(session.prefill(prompt, SamplingLogits::Full, &mut |_| {})?.logits)?;
    let mut trace = Trace {
        tokens: Vec::new(),
        margins: Vec::new(),
        decode: Duration::ZERO,
    };
    if let Some(barrier) = barrier {
        barrier.wait();
    }
    let started = Instant::now();
    for step in 0..steps {
        let (token, margin) = top_two(&logits)?;
        trace.tokens.push(token);
        trace.margins.push(margin);
        if step + 1 < steps {
            if let Some(barrier) = barrier {
                barrier.wait();
            }
            logits = full(session.decode(token, SamplingLogits::Full)?.logits)?;
        }
    }
    trace.decode = started.elapsed();
    Ok(trace)
}

fn full(trace: Option<runtime::backend::LogitsTrace>) -> Result<Vec<f32>, Failure> {
    Ok(trace.ok_or("backend did not return full logits")?.values)
}

/// The argmax and its margin over the runner-up.
fn top_two(values: &[f32]) -> Result<(u32, f32), Failure> {
    let (mut best, mut first, mut second) = (0, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (index, &value) in values.iter().enumerate() {
        if value > first {
            (best, second, first) = (index, first, value);
        } else if value > second {
            second = value;
        }
    }
    Ok((u32::try_from(best)?, first - second))
}
