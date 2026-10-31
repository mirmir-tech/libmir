//! Repeats the prefill of one reference prompt in fresh sessions of one
//! loaded model, with timing-based kernel selection disabled, and compares
//! the next-token logits of every repeat with the first. Later repeats may
//! reuse the prefix cache the first one published; `RUST_LOG=libmir=debug`
//! shows the cached token count of each.
//!
//! Usage: `prefill-determinism <model> <reference.json> [record] [repeats]
//! [chunk_tokens] [loads] [kv_blocks]`, with prompts from
//! `benchmarks/tools/nemotron/greedy_oracle.py`. With several loads, the model
//! is unloaded and loaded again in the same process, so each load's first
//! repeat is a fresh prefill compared with the first load's. Prints a digest
//! of the first logits so separate processes can be compared.
#![allow(clippy::print_stdout)]

use std::{env, fs, path::PathBuf};

use libmir::{GenerationOverrides, Library, RuntimeConfig, SamplingLogits};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    prompt_ids: Vec<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init(),
    );
    let mut args = env::args().skip(1);
    let model = args.next().map(PathBuf::from).ok_or("missing model path")?;
    let reference: Reference =
        serde_json::from_slice(&fs::read(args.next().ok_or("missing reference path")?)?)?;
    let record = args.next().map_or(Ok(0), |value| value.parse())?;
    let repeats = args.next().map_or(Ok(3), |value| value.parse::<usize>())?;
    let chunk = args.next().map(|value| value.parse::<usize>()).transpose()?;
    let loads = args.next().map_or(Ok(1), |value| value.parse::<usize>())?;
    let kv_blocks = args.next().map(|value| value.parse::<u32>()).transpose()?;
    let prompt = &reference.records.get(record).ok_or("no such record")?.prompt_ids;
    #[allow(unused_mut)]
    let mut config = RuntimeConfig::default();
    if let Some(blocks) = kv_blocks {
        config.kv_cache.block_count = blocks;
        config.automatic_kv_cache = false;
    }
    #[cfg(feature = "cuda")]
    {
        config.cuda.tuning.mode = runtime::tuning::TuningMode::Disabled;
        if let Some(chunk) = chunk {
            config.cuda.model_session.prefill_chunk_tokens = chunk;
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = chunk;
    let mut first: Option<Vec<f32>> = None;
    for load in 0..loads {
        let model = Library::new(config.clone()).load(
            &model,
            GenerationOverrides::default(),
            &mut |_| {},
        )?;
        compare(&model, prompt, load, repeats, &mut first)?;
        model.unload()?;
    }
    Ok(())
}

fn compare(
    model: &libmir::Model,
    prompt: &[u32],
    load: usize,
    repeats: usize,
    first: &mut Option<Vec<f32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    for repeat in 0..repeats {
        let mut session = model.session();
        let logits = session
            .prefill(prompt, SamplingLogits::Full, &mut |_| {})?
            .logits
            .ok_or("backend did not return full logits")?
            .values;
        if repeat == 0 {
            let digest = logits.iter().map(|value| f64::from(*value)).sum::<f64>();
            println!(
                "prompt_tokens={} load={load} repeat=0 argmax={} digest={digest:.6}",
                prompt.len(),
                argmax(&logits)
            );
        }
        let Some(first) = first.as_ref() else {
            *first = Some(logits);
            continue;
        };
        let difference = first
            .iter()
            .zip(&logits)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "load={load} repeat={repeat} argmax={} max_abs_logit_difference={difference:.4}",
            argmax(&logits)
        );
    }
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().fold(0, |best, (index, value)| {
        if *value > values[best] {
            index
        } else {
            best
        }
    })
}
