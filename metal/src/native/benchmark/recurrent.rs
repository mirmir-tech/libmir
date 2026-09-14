use std::{io::Write, time::Instant};

use models::layout::AttentionLayerType;
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{BenchmarkConfig, greedy_token};
use crate::native::{
    error::Result,
    model::{DecodeInput, LoadedModel},
};

#[test]
#[ignore = "loads a real hybrid model; set MIRMIR_BENCH_MODEL"]
fn compares_complete_decode_with_reused_and_concatenated_gdn_state() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let context = super::positive_env("MIRMIR_BENCH_CONTEXT", 2048)?;
    let batch = super::positive_env("MIRMIR_BENCH_BATCH", 5)?;
    let mut metal = super::diagnostics::isolated_config();
    let settings = std::sync::Arc::make_mut(&mut metal);
    settings.cache.prefix_cache_entries = 0;
    let rows = batch.checked_mul(2).ok_or(crate::engine::Error::ShapeOverflow)?;
    let tokens = context
        .checked_add(config.decode_tokens)
        .and_then(|value| value.checked_add(1))
        .ok_or(crate::engine::Error::ShapeOverflow)?;
    let blocks = tokens
        .div_ceil(settings.kv_cache.block_size)
        .checked_mul(rows)
        .ok_or(crate::engine::Error::ShapeOverflow)?;
    settings.kv_cache.block_count = settings.kv_cache.block_count.max(u32::try_from(blocks)?);
    settings.set_max_batch_requests(rows);
    let mut model = LoadedModel::load_with_config(&config.manifest(), metal, &mut |_| {})?;
    let decoder = model
        .info
        .decoder
        .as_ref()
        .ok_or_else(|| super::Error::Benchmark("missing decoder configuration".into()))?;
    let layers = (0..decoder.num_hidden_layers)
        .filter(|index| decoder.layer_type(*index) == AttentionLayerType::Linear)
        .collect::<Vec<_>>();
    assert!(!layers.is_empty());
    let mut cohorts = [Vec::new(), Vec::new()];
    for cohort in &mut cohorts {
        for row in 0..batch {
            let prompt = (0..context)
                .map(|index| Ok(u32::try_from((row * 1009 + index) % 100_000 + 1000)?))
                .collect::<Result<Vec<_>>>()?;
            let session = Uuid::new_v4();
            let output =
                model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
            cohort.push(DecodeInput {
                session,
                token: greedy_token(&output.output)?,
                sampling: SamplingLogits::None,
            });
        }
    }
    model.stream().synchronize()?;
    let mut timings = [Vec::new(), Vec::new()];
    let mut digest = blake3::Hasher::new();
    for step in 0..config.decode_tokens {
        let order = if step % 4 < 2 {
            [0, 1]
        } else {
            [1, 0]
        };
        for variant in order {
            if variant == 1 {
                for input in &cohorts[variant] {
                    let state = model
                        .sessions
                        .get_mut(&input.session)
                        .ok_or_else(|| super::Error::Benchmark("missing decode session".into()))?;
                    for layer in &layers {
                        state.cache.gated_delta_state(*layer)?.discard_packed_source()?;
                    }
                }
            }
            let started = Instant::now();
            let outputs = model.decode_batch(&cohorts[variant])?;
            model.stream().synchronize()?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            // First steps populate both shape variants and are excluded from timing.
            if step >= 8 {
                timings[variant].push(elapsed);
            }
            for (input, output) in cohorts[variant].iter_mut().zip(outputs) {
                input.token = greedy_token(&output)?;
            }
        }
        for (row, (a, b)) in cohorts[0].iter().zip(&cohorts[1]).enumerate() {
            assert_eq!(a.token, b.token, "GDN storage row {row}, step {step}");
            digest.update(&a.token.to_le_bytes());
        }
    }
    report_timings(&mut timings, context, batch)?;
    writeln!(std::io::stderr().lock(), "gdn_storage.digest: {}", digest.finalize())?;
    for input in cohorts.into_iter().flatten() {
        model.release_session(input.session)?;
    }
    Ok(())
}

fn report_timings(timings: &mut [Vec<f64>; 2], context: usize, batch: usize) -> Result<()> {
    let mut ratios = timings[1]
        .iter()
        .zip(&timings[0])
        .map(|(baseline, reuse)| baseline / reuse)
        .collect::<Vec<_>>();
    ratios.sort_by(f64::total_cmp);
    assert!(!ratios.is_empty(), "decode requires more than eight steps");
    writeln!(
        std::io::stderr().lock(),
        "gdn_storage.paired: median_speedup={:.5}",
        ratios[ratios.len() / 2]
    )?;
    for (name, samples) in ["reuse", "concatenate"].into_iter().zip(timings.iter_mut()) {
        samples.sort_by(f64::total_cmp);
        assert!(!samples.is_empty(), "decode requires more than eight steps");
        writeln!(
            std::io::stderr().lock(),
            "gdn_storage.benchmark: variant={name}, context={context}, batch={batch}, samples={}, median_ms={:.3}, mean_ms={:.3}",
            samples.len(),
            samples[samples.len() / 2],
            samples.iter().sum::<f64>() / f64::from(u32::try_from(samples.len())?),
        )?;
        writeln!(
            std::io::stderr().lock(),
            "gdn_storage.samples: variant={name}, milliseconds={samples:?}"
        )?;
    }
    Ok(())
}
