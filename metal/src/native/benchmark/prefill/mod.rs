mod diagnosis;
mod memory;

use std::{env, io::Write, time::Instant};

use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{BenchmarkConfig, Result, diagnostics, greedy_token, positive_env};
use crate::native::model::LoadedModel;

#[test]
#[ignore = "loads a real model and compares prefill chunks; set MIRMIR_BENCH_MODEL"]
fn measures_prefill_chunk_candidates() -> Result<()> {
    let mut config = BenchmarkConfig::from_env()?;
    let context = positive_env("MIRMIR_BENCH_CONTEXT", 2_048)?;
    config.prompt_tokens = context;
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load_with_config(
        &config.manifest(),
        diagnostics::isolated_config(),
        &mut ignored,
    )?;
    let mut report = std::io::stderr().lock();
    let mut mismatches = 0;
    let steps = env::var("MIRMIR_BENCH_PREFILL_STEPS")
        .unwrap_or_else(|_| "512,1024,2048".into())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    assert!(steps.iter().all(|step| *step > 0));
    for round in 0..config.warmup + config.samples {
        let prompt = (0..context)
            .map(|index| Ok(u32::try_from((index + round * 1_009) % 100_000 + 1_000)?))
            .collect::<Result<Vec<_>>>()?;
        let mut order = steps.clone();
        if !round.is_multiple_of(2) {
            order.reverse();
        }
        let mut reference = None;
        for chunk in order {
            model.info.prefill_step = chunk;
            model.clear_prefix_cache();
            crate::engine::clear_memory_cache()?;
            let session = Uuid::new_v4();
            let started = Instant::now();
            let output =
                model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut ignored)?;
            let prefill_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let mut tokens = vec![greedy_token(&output.output)?];
            for _ in 0..config.decode_tokens {
                let output =
                    model.decode(session, tokens[tokens.len() - 1], SamplingLogits::None)?;
                tokens.push(greedy_token(&output)?);
            }
            let memory = crate::engine::memory_stats()?;
            let agrees = reference.as_ref().is_none_or(|expected| expected == &tokens);
            if !agrees {
                mismatches += 1;
            }
            if reference.is_none() {
                reference = Some(tokens.clone());
            }
            writeln!(
                report,
                "prefill_candidate: {}",
                serde_json::json!({
                    "round": round, "measured": round >= config.warmup, "context": context,
                    "chunk": chunk, "prefill_ms": prefill_ms,
                "initial_chunk": model.prefill_chunk_len(0, context.saturating_sub(1)),
                    "active_bytes": memory.active, "peak_bytes_process": memory.peak,
                    "matches_first_candidate": agrees, "tokens": tokens,
                })
            )?;
            model.release_session(session)?;
        }
    }
    assert_eq!(mismatches, 0, "prefill chunk candidates changed the greedy trajectory");
    Ok(())
}
