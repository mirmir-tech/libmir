mod aligned;
mod model;
mod schedule;
use std::{io::Write, sync::Arc, time::Instant};

use runtime::{
    backend::{ModelHandle, PrefillRequest, SamplingLogits},
    kv::BlockTable,
    tuning::TuningMode,
};
use schedule::Schedule;
use serde_json::json;
use uuid::Uuid;

use super::super::{BenchmarkConfig, diagnostics, greedy_token};
use crate::{
    config::{GatedDeltaPrefill, PrefixRetention},
    native::{
        error::Result,
        model::{DecodeInput, LoadedModel},
        prefill::MetalPrefillBatch,
    },
};

#[test]
#[ignore = "fixed-plan repeated C5 prefill diagnosis; set MIRMIR_BENCH_MODEL"]
fn diagnoses_repeated_packed_prefill() -> Result<()> {
    diagnose(Schedule::Adaptive, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}

#[test]
#[ignore = "C5 compaction isolation with identical 128-token chunks; set MIRMIR_BENCH_MODEL"]
fn diagnoses_uniform_chunk_prefill() -> Result<()> {
    diagnose(Schedule::Uniform128, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}

#[test]
#[ignore = "C5 checkpoint-only compaction at identical 128-token chunks; set MIRMIR_BENCH_MODEL"]
fn diagnoses_uniform_checkpoint_retention() -> Result<()> {
    diagnose(
        Schedule::Uniform128,
        PrefixRetention::CompactCheckpoint,
        None,
        GatedDeltaPrefill::Rows,
    )
}

#[test]
#[ignore = "profiles two warm C5/128 prefill chunks; set MIRMIR_BENCH_MODEL"]
fn diagnoses_windowed_prefill_components() -> Result<()> {
    diagnose(
        Schedule::Uniform128,
        PrefixRetention::View,
        Some(7680..7936),
        GatedDeltaPrefill::Rows,
    )
}

#[test]
#[ignore = "packed Gated Delta prefill candidate; set MIRMIR_BENCH_MODEL"]
fn diagnoses_packed_gated_delta_prefill() -> Result<()> {
    diagnose(Schedule::Uniform128, PrefixRetention::View, None, GatedDeltaPrefill::Packed)
}

#[test]
#[ignore = "first-chunk projection diagnosis on actual Gated Delta weights; set MIRMIR_BENCH_MODEL"]
fn diagnoses_gated_delta_prefill_projections() -> Result<()> {
    diagnose(Schedule::Probe, PrefixRetention::View, None, GatedDeltaPrefill::DiagnosePacked)
}

#[test]
#[ignore = "same-process warmed GDN ABBA on actual weights; set MIRMIR_BENCH_MODEL"]
fn compares_warm_gated_delta_prefill() -> Result<()> {
    diagnose(Schedule::Paired, PrefixRetention::View, None, GatedDeltaPrefill::ComparePacked)
}

#[test]
#[ignore = "warmed actual MoE routing and component diagnosis; set MIRMIR_BENCH_MODEL"]
fn compares_warm_moe_prefill() -> Result<()> {
    diagnose(Schedule::MoePaired, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}

fn diagnose(
    schedule: Schedule,
    retention: PrefixRetention,
    window: Option<std::ops::Range<usize>>,
    gdn: GatedDeltaPrefill,
) -> Result<()> {
    initialize_tracing();
    let mut config = BenchmarkConfig::from_env()?;
    config.prompt_tokens = 8192;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.diagnostics.prefix_retention = retention;
    metal.diagnostics.prefill_component_window = window;
    metal.diagnostics.gated_delta_prefill = gdn;
    metal.diagnostics.moe_prefill = schedule.moe_diagnostic();
    metal.set_max_batch_requests(5);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let (contexts, budget) = schedule.configure(&mut model);
    for (round, &context) in contexts.iter().enumerate() {
        let requests = (0..5)
            .map(|row| request(&model, round, row, context))
            .collect::<Result<Vec<_>>>()?;
        let started = Instant::now();
        let (batch, _) = MetalPrefillBatch::prepare(&mut model, requests, None)?;
        let mut complete = false;
        let mut progress = Vec::new();
        for iteration in 0..512 {
            let step = batch.execute_step(&mut model, budget)?;
            let positions = step
                .events
                .iter()
                .map(|(row, event)| (*row, event.count().current()))
                .collect::<Vec<_>>();
            if !matches!(schedule, Schedule::Adaptive | Schedule::Probe) && !step.complete {
                assert_eq!(positions.len(), 5, "memory fallback invalidates uniform scheduling");
                for &(row, position) in &positions {
                    assert!(row < 5);
                    assert_eq!(position, u64::try_from((iteration + 1) * model.info.prefill_step)?);
                }
            }
            progress.push(positions.clone());
            writeln!(
                std::io::stderr().lock(),
                "prefill.progress: {}",
                json!({
                    "round": round, "context": context, "iteration": iteration,
                    "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
                    "positions": positions,
                })
            )?;
            if step.complete {
                complete = true;
                break;
            }
        }
        assert!(complete, "bounded diagnostic prefill must complete");
        let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut inputs = batch
            .finish()?
            .into_iter()
            .map(|finished| {
                Ok(DecodeInput {
                    session: finished.request.session_id,
                    token: greedy_token(&finished.native.output)?,
                    sampling: SamplingLogits::None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tokens = vec![Vec::new(); inputs.len()];
        let started = Instant::now();
        for _ in 0..16 {
            for (row, input) in inputs.iter().enumerate() {
                tokens[row].push(input.token);
            }
            let outputs = model.decode_batch(&inputs)?;
            for (input, output) in inputs.iter_mut().zip(outputs) {
                input.token = greedy_token(&output)?;
            }
        }
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
        for input in inputs {
            model.release_session(input.session)?;
        }
        let memory = crate::engine::memory_stats()?;
        writeln!(
            std::io::stderr().lock(),
            "prefill.diagnosis: {}",
            json!({
                "round": round, "context": context, "prefill_ms": prefill_ms,
                "decode_ms": decode_ms, "tokens": tokens,
                "active_bytes": memory.active, "cached_bytes": memory.cached,
                "prefix_groups": model.prefixes.group_count(),
                "prefix_bytes": model.prefixes.resident_bytes(),
                "recurrent_bytes": model.prefixes.recurrent_bytes(),
                "convolution_bytes": model.prefixes.convolution_bytes()?,
                "schedule": progress,
            })
        )?;
    }
    Ok(())
}

fn initialize_tracing() {
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_ansi(false)
            .try_init(),
    );
}

fn request(
    model: &LoadedModel,
    round: usize,
    row: usize,
    context: usize,
) -> Result<(PrefillRequest, SamplingLogits)> {
    let prompt_tokens = (0..context)
        .map(|index| Ok(u32::try_from((round * 5003 + row * 1009 + index) % 100_000 + 1000)?))
        .collect::<Result<Vec<_>>>()?;
    Ok((
        PrefillRequest {
            model: ModelHandle {
                id: model.info.manifest.id.clone(),
                backend: "metal".into(),
            },
            session_id: Uuid::new_v4(),
            prompt_tokens,
            cache_checkpoints: (2048..context).step_by(2048).collect(),
            block_table: BlockTable::with_block_size(16),
            cached_tokens: 0,
            generation_tokens: None,
            sampling_logits: SamplingLogits::None,
        },
        SamplingLogits::None,
    ))
}

#[test]
#[ignore = "warmed actual MXFP4 expert projection fusion; set MIRMIR_BENCH_MODEL"]
fn compares_warm_moe_projection() -> Result<()> {
    diagnose(Schedule::MoeProjection, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}

#[test]
#[ignore = "warmed actual MXFP4 indexed input gather; set MIRMIR_BENCH_MODEL"]
fn compares_warm_moe_indexed() -> Result<()> {
    diagnose(Schedule::MoeIndexed, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}

#[test]
#[ignore = "separate numerical and timing diagnosis of indexed MoE; set MIRMIR_BENCH_MODEL"]
fn measures_indexed_moe_numerics() -> Result<()> {
    diagnose(
        Schedule::MoeIndexedNumerics,
        PrefixRetention::View,
        None,
        GatedDeltaPrefill::Rows,
    )
}
