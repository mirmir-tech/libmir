use std::{io::Write, sync::Arc, time::Instant};

use runtime::{backend::SamplingLogits, tuning::TuningMode};
use uuid::Uuid;

use super::{BenchmarkConfig, output_token};
use crate::{
    engine::BatchAttentionExecution,
    native::{
        error::Result,
        model::{DecodeInput, LoadedModel},
    },
};

#[test]
#[ignore = "whole-model paged batch attention candidate; set MIRMIR_BENCH_MODEL"]
fn compares_paged_attention_candidate_with_row_reader() -> Result<()> {
    let context = super::super::positive_env("MIRMIR_BENCH_PIPELINE_CONTEXT", 8193)?;
    let rows = super::super::positive_env("MIRMIR_BENCH_BATCH", 5)?;
    let candidate = candidate(context, rows)?;
    let config = BenchmarkConfig {
        model: super::super::model_path()?,
        prompt_tokens: context,
        decode_tokens: 128,
        samples: 1,
        warmup: 0,
    };
    let mut settings = super::super::diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 0;
    metal.set_max_batch_requests(rows * 2);
    let blocks = (context + 129).div_ceil(metal.kv_cache.block_size) * rows * 2;
    metal.kv_cache.block_count = metal.kv_cache.block_count.max(u32::try_from(blocks)?);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let mut cohorts = [Vec::new(), Vec::new()];
    for row in 0..rows {
        let prompt = (0..context)
            .map(|index| Ok(u32::try_from((row * 1009 + index) % 100_000 + 1000)?))
            .collect::<Result<Vec<_>>>()?;
        let session = Uuid::new_v4();
        let output =
            model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
        let token = output_token(&model, output.output)?;
        let cloned = model.sessions[&session].snapshot()?;
        let reference = Uuid::new_v4();
        model.sessions.insert(reference, cloned);
        cohorts[0].push(DecodeInput {
            session,
            token,
            sampling: SamplingLogits::None,
        });
        cohorts[1].push(DecodeInput {
            session: reference,
            token,
            sampling: SamplingLogits::None,
        });
        writeln!(std::io::stderr().lock(), "attention.prefill: row={row}, context={context}")?;
    }
    let mut times = [Vec::new(), Vec::new()];
    let mut digest = blake3::Hasher::new();
    for step in 0..128 {
        for variant in if step % 4 < 2 {
            [0, 1]
        } else {
            [1, 0]
        } {
            model.stream.set_attention_candidate((variant == 0).then_some(candidate));
            let started = Instant::now();
            let outputs = if rows > 1 {
                model.decode_batch(&cohorts[variant])?
            } else {
                let input = cohorts[variant][0];
                vec![model.decode(input.session, input.token, input.sampling)?]
            };
            for (input, output) in cohorts[variant].iter_mut().zip(outputs) {
                input.token = output_token(&model, output)?;
            }
            model.stream().synchronize()?;
            if step >= 8 {
                times[variant].push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        for (row, (a, b)) in cohorts[0].iter().zip(&cohorts[1]).enumerate() {
            assert_eq!(a.token, b.token, "attention candidate row {row}, step {step}");
            digest.update(&a.token.to_le_bytes());
        }
    }
    let mut ratios = times[1].iter().zip(&times[0]).map(|(b, a)| b / a).collect::<Vec<_>>();
    ratios.sort_by(f64::total_cmp);
    writeln!(
        std::io::stderr().lock(),
        "attention.paired: context={context}, rows={rows}, candidate={candidate:?}, median_speedup={:.5}, digest={}",
        ratios[ratios.len() / 2],
        digest.finalize()
    )?;
    for (name, mut samples) in ["candidate", "rows"].into_iter().zip(times) {
        writeln!(std::io::stderr().lock(), "attention.samples: {name}={samples:?}")?;
        samples.sort_by(f64::total_cmp);
        writeln!(
            std::io::stderr().lock(),
            "attention.median: {name}={:.3}",
            samples[samples.len() / 2]
        )?;
    }
    model.stream.set_attention_candidate(None);
    for input in cohorts.into_iter().flatten() {
        model.release_session(input.session)?;
    }
    Ok(())
}

fn candidate(context: usize, rows: usize) -> Result<BatchAttentionExecution> {
    if context < 8193 || rows < 2 {
        return Err(super::super::Error::Benchmark(
            "paged candidate requires context >= 8193 and batch >= 2".into(),
        ));
    }
    let width = super::super::positive_env("MIRMIR_BENCH_ATTENTION_ROWS", 12)?;
    match (std::env::var("MIRMIR_BENCH_ATTENTION_PLAN").as_deref(), width) {
        (Ok("two-pass"), _) => Ok(BatchAttentionExecution::PagedBatchedTwoPass),
        (_, 4) => Ok(BatchAttentionExecution::PagedBatched4),
        (_, 8) => Ok(BatchAttentionExecution::PagedBatched8),
        (_, 12) => Ok(BatchAttentionExecution::PagedBatched12),
        _ => Err(super::super::Error::Benchmark("candidate width must be 4, 8 or 12".into())),
    }
}
