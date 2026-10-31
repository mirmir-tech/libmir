mod chunks;
mod diagnosis;
mod layers;
mod state;

use std::{io::Write, sync::Arc, time::Instant};

use runtime::{backend::SamplingLogits, tuning::TuningMode};
use serde_json::json;
use uuid::Uuid;

use super::{BenchmarkConfig, LoadedModel, Result, diagnostics, greedy_token, positive_env};
use crate::config::PrefillOutput::{self, DeferredToken, LastChunk, SeparateToken};

#[derive(serde::Serialize)]
struct Observation {
    start_epoch_ms: u128,
    prefill_end_epoch_ms: u128,
    end_epoch_ms: u128,
    prefill_ms: f64,
    decode_ms: f64,
    tokens: Vec<u32>,
    progress: Vec<u64>,
    active_bytes: usize,
    cached_bytes: usize,
    peak_bytes_process: usize,
}

#[test]
#[ignore = "bounded C1 terminal-prefill comparison on one MLX version; set MIRMIR_BENCH_MODEL"]
fn compares_terminal_prefill() -> Result<()> {
    compare(LastChunk)
}

#[test]
#[ignore = "bounded C1 deferred terminal synchronization comparison; set MIRMIR_BENCH_MODEL"]
fn compares_deferred_prefill() -> Result<()> {
    compare(DeferredToken)
}

fn compare(candidate: PrefillOutput) -> Result<()> {
    let mut config = BenchmarkConfig::from_env()?;
    let context = positive_env("MIRMIR_BENCH_CONTEXT", 2065)?;
    config.prompt_tokens = context;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 4;
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    model.info.prefill_step = 512;
    let prompt = (0..context)
        .map(|index| Ok(u32::try_from(index % 100_000 + 1_000)?))
        .collect::<Result<Vec<_>>>()?;
    let checkpoint = context.saturating_sub(16) / 16 * 16;
    let mut reference = None;
    let mut samples = Vec::new();
    // One warm-up ABBA block, followed by ABBA and BAAB. Never retry or
    // omit an observation after a failed numerical or stability gate.
    for (block, order) in [
        [SeparateToken, candidate, candidate, SeparateToken],
        [SeparateToken, candidate, candidate, SeparateToken],
        [candidate, SeparateToken, SeparateToken, candidate],
    ]
    .into_iter()
    .enumerate()
    {
        for plan in order {
            model.stream.set_prefill_output(plan);
            let observation = run(&mut model, &prompt, checkpoint, config.decode_tokens)?;
            let agrees = reference.as_ref().is_none_or(|tokens| tokens == &observation.tokens);
            writeln!(
                std::io::stderr().lock(),
                "terminal.sample: {}",
                json!({
                    "mlx": mirtal::version().map_err(crate::engine::Error::from)?,
                    "context": context, "checkpoint": checkpoint, "block": block,
                    "plan": plan, "measured": block > 0, "matches_reference": agrees,
                    "observation": observation,
                })
            )?;
            assert!(
                agrees,
                "terminal prefill changed the greedy trajectory; no timing qualification"
            );
            if reference.is_none() {
                reference = Some(observation.tokens.clone());
            }
            if block > 0 {
                samples.push((plan, observation.prefill_ms, observation.decode_ms));
                for mode in [SeparateToken, candidate] {
                    for values in [
                        samples.iter().filter(|s| s.0 == mode).map(|s| s.1).collect::<Vec<_>>(),
                        samples.iter().filter(|s| s.0 == mode).map(|s| s.2).collect::<Vec<_>>(),
                    ] {
                        if values.len() > 1 {
                            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                            let max = values.iter().copied().fold(0.0, f64::max);
                            assert!(max <= min * 1.15, "unstable terminal-prefill control; stop");
                        }
                    }
                }
            }
        }
    }
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(())
}

fn run(
    model: &mut LoadedModel,
    prompt: &[u32],
    checkpoint: usize,
    decode_tokens: usize,
) -> Result<Observation> {
    model.clear_prefix_cache();
    model.stream.synchronize()?;
    let session = Uuid::new_v4();
    let mut progress = Vec::new();
    let start_epoch_ms = epoch_ms()?;
    let started = Instant::now();
    let output = model.prefill(
        session,
        prompt,
        &[checkpoint],
        &SamplingLogits::None,
        Some(16),
        &mut |event| progress.push(event.count().current()),
    )?;
    model.stream.synchronize()?;
    let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
    let prefill_end_epoch_ms = epoch_ms()?;
    let mut tokens = vec![greedy_token(&output.output)?];
    let started = Instant::now();
    for _ in 1..decode_tokens {
        let output = model.decode(session, tokens[tokens.len() - 1], SamplingLogits::None)?;
        tokens.push(greedy_token(&output)?);
    }
    model.stream.synchronize()?;
    let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
    let end_epoch_ms = epoch_ms()?;
    let memory = crate::engine::memory_stats()?;
    model.release_session(session)?;
    Ok(Observation {
        start_epoch_ms,
        prefill_end_epoch_ms,
        end_epoch_ms,
        prefill_ms,
        decode_ms,
        tokens,
        progress,
        active_bytes: memory.active,
        cached_bytes: memory.cached,
        peak_bytes_process: memory.peak,
    })
}

fn epoch_ms() -> Result<u128> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .map_err(|error| crate::native::error::Error::Benchmark(error.to_string()))
}
