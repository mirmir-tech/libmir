use std::{io::Write, sync::Arc, time::Instant};

use runtime::{backend::SamplingLogits, tuning::TuningMode};
use uuid::Uuid;

use super::{BenchmarkConfig, DECODE_TOKENS, output_token};
use crate::native::{
    error::Result,
    model::{DecodeExecution, DecodeInput, LoadedModel},
};

#[test]
#[ignore = "loads a real model and compares mixed decode groups; set MIRMIR_BENCH_MODEL"]
fn compares_mixed_grouped_decode_against_scalar_fallback() -> Result<()> {
    let context = super::super::positive_env("MIRMIR_BENCH_PIPELINE_CONTEXT", 128)?;
    let config = BenchmarkConfig {
        model: super::super::model_path()?,
        decode_tokens: DECODE_TOKENS,
        prompt_tokens: context,
        samples: 1,
        warmup: 0,
    };
    let mut settings = super::super::diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 0;
    let blocks = (context + DECODE_TOKENS + 1).div_ceil(metal.kv_cache.block_size) * 10;
    metal.kv_cache.block_count = metal.kv_cache.block_count.max(u32::try_from(blocks)?);
    metal.set_max_batch_requests(10);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let mut cohorts = [Vec::new(), Vec::new()];
    for cohort in &mut cohorts {
        for row in 0..5 {
            let prompt = (0..context)
                .map(|index| Ok(u32::try_from((row * 1009 + index) % 100_000 + 1000)?))
                .collect::<Result<Vec<_>>>()?;
            let session = Uuid::new_v4();
            let output =
                model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
            cohort.push(DecodeInput {
                session,
                token: output_token(&model, output.output)?,
                sampling: SamplingLogits::None,
            });
        }
    }
    let mut timings = [Vec::new(), Vec::new()];
    let mut digest = blake3::Hasher::new();
    let mut packed_outputs = 0;
    for step in 0..DECODE_TOKENS {
        // A different row requests full logits every 32 steps, then returns
        // to the device pipeline. Both variants use the same scalar policy.
        for cohort in &mut cohorts {
            for (row, input) in cohort.iter_mut().enumerate() {
                input.sampling = if row == step / 32 {
                    SamplingLogits::Full
                } else {
                    SamplingLogits::None
                };
            }
        }
        for variant in if step % 4 < 2 {
            [0, 1]
        } else {
            [1, 0]
        } {
            let started = Instant::now();
            let outputs = if variant == 0 {
                model.decode_grouped(&cohorts[variant])?
            } else {
                cohorts[variant]
                    .iter()
                    .map(|input| {
                        Ok((
                            model.decode(input.session, input.token, input.sampling)?,
                            DecodeExecution::Scalar,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            for (input, (output, execution)) in cohorts[variant].iter_mut().zip(outputs) {
                input.token = output_token(&model, output)?;
                packed_outputs += usize::from(matches!(execution, DecodeExecution::Packed { .. }));
            }
            model.stream().synchronize()?;
            if step >= 8 {
                timings[variant].push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        for (row, (a, b)) in cohorts[0].iter().zip(&cohorts[1]).enumerate() {
            assert_eq!(a.token, b.token, "mixed cohort row {row}, step {step}");
            digest.update(&a.token.to_le_bytes());
        }
    }
    assert!(packed_outputs > DECODE_TOKENS * 3);
    let mut ratios = timings[1].iter().zip(&timings[0]).map(|(b, a)| b / a).collect::<Vec<_>>();
    ratios.sort_by(f64::total_cmp);
    writeln!(
        std::io::stderr().lock(),
        "mixed_decode.paired: context={context}, median_speedup={:.5}, digest={}",
        ratios[ratios.len() / 2],
        digest.finalize()
    )?;
    for (variant, mut samples) in ["grouped", "scalar"].into_iter().zip(timings) {
        writeln!(std::io::stderr().lock(), "mixed_decode.samples: {variant}={samples:?}")?;
        samples.sort_by(f64::total_cmp);
        writeln!(
            std::io::stderr().lock(),
            "mixed_decode.median: variant={variant}, ms={:.3}",
            samples[samples.len() / 2]
        )?;
    }
    for input in cohorts.into_iter().flatten() {
        model.release_session(input.session)?;
    }
    Ok(())
}
