mod decode;
mod drift;
mod run;
mod settled;
mod tiles;
use run::{run_budget_with_decode, run_width_with_decode};

use super::*;
use crate::config::MoePrefill;

#[derive(Debug, serde::Serialize)]
struct Observation {
    prefill_ms: f64,
    decode_ms: f64,
    tokens: Vec<Vec<u32>>,
    schedule: Vec<Vec<(usize, u64)>>,
    active_bytes: usize,
    cached_bytes: usize,
    peak_bytes_process: usize,
}

#[test]
#[ignore = "same-model aligned MoE C5 full-prefill qualification; set MIRMIR_BENCH_MODEL"]
fn compares_aligned_moe_full_model() -> Result<()> {
    initialize_tracing();
    let mut config = BenchmarkConfig::from_env()?;
    config.prompt_tokens = 8193;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.set_max_batch_requests(5);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    model.info.prefill_step = 512;
    let mut reference = None;
    let mut timings = Vec::new();
    // Both variants warm the same allocator and kernels before the ABBA.
    for block in 0..4 {
        for (slot, mode) in
            [MoePrefill::Default, MoePrefill::Aligned, MoePrefill::Aligned, MoePrefill::Default]
                .into_iter()
                .enumerate()
        {
            let observation = run(&mut model, mode, 2049)?;
            let matches = reference.as_ref().is_none_or(|prior: &Observation| {
                prior.tokens == observation.tokens && prior.schedule == observation.schedule
            });
            writeln!(
                std::io::stderr().lock(),
                "moe.model: {}",
                json!({
                    "context": 2049, "block": block, "slot": slot, "measured": block > 0,
                    "plan": name(mode), "matches_reference": matches, "observation": observation,
                })
            )?;
            assert!(matches, "full-model candidate changed tokens or schedule");
            if block > 0 {
                timings.push((mode, observation.prefill_ms, observation.decode_ms));
                assert!(
                    stable(&timings),
                    "unstable prefill/decode control; abort timing qualification"
                );
            }
            if reference.is_none() {
                reference = Some(observation);
            }
        }
    }
    let reference = run(&mut model, MoePrefill::Default, 8193)?;
    writeln!(
        std::io::stderr().lock(),
        "moe.long: {}",
        json!({"plan": "grouped_fused", "observation": reference})
    )?;
    let candidate = run(&mut model, MoePrefill::Aligned, 8193)?;
    writeln!(
        std::io::stderr().lock(),
        "moe.long: {}",
        json!({"plan": "grouped_aligned", "observation": candidate})
    )?;
    assert_eq!(reference.tokens, candidate.tokens, "long-context greedy parity");
    assert_eq!(reference.schedule, candidate.schedule, "long-context scheduling parity");
    Ok(())
}

#[test]
#[ignore = "semantic-only aligned MoE C5/8193 gate; set MIRMIR_BENCH_MODEL"]
fn preserves_aligned_moe_long_context() -> Result<()> {
    initialize_tracing();
    let mut config = BenchmarkConfig::from_env()?;
    config.prompt_tokens = 8193;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.set_max_batch_requests(5);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    model.info.prefill_step = 512;
    let reference = run(&mut model, MoePrefill::Default, 8193)?;
    writeln!(
        std::io::stderr().lock(),
        "moe.long: {}",
        json!({"plan": "grouped_fused", "observation": reference})
    )?;
    let candidate = run(&mut model, MoePrefill::Aligned, 8193)?;
    writeln!(
        std::io::stderr().lock(),
        "moe.long: {}",
        json!({"plan": "grouped_aligned", "observation": candidate})
    )?;
    assert_eq!(reference.tokens, candidate.tokens, "long-context greedy parity");
    assert_eq!(reference.schedule, candidate.schedule, "long-context scheduling parity");
    Ok(())
}

fn run(model: &mut LoadedModel, mode: MoePrefill, context: usize) -> Result<Observation> {
    run_with_decode(model, mode, context, decode::plain)
}

fn run_with_decode(
    model: &mut LoadedModel,
    mode: MoePrefill,
    context: usize,
    decode: impl FnOnce(&mut LoadedModel, &mut [DecodeInput]) -> Result<decode::Observation>,
) -> Result<Observation> {
    run_width_with_decode(model, mode, context, 5, decode)
}

fn name(mode: MoePrefill) -> &'static str {
    match mode {
        MoePrefill::Default => "grouped_fused",
        MoePrefill::Aligned => "grouped_aligned",
        MoePrefill::Tiles64 => "grouped_tiles64",
        _ => unreachable!(),
    }
}

fn stable(samples: &[(MoePrefill, f64, f64)]) -> bool {
    [MoePrefill::Default, MoePrefill::Aligned, MoePrefill::Tiles64]
        .into_iter()
        .all(|mode| {
            let rows = samples.iter().filter(|s| s.0 == mode).collect::<Vec<_>>();
            if rows.len() < 2 {
                return true;
            }
            let prefill = rows.iter().map(|s| s.1).collect::<Vec<_>>();
            let decode = rows.iter().map(|s| s.2).collect::<Vec<_>>();
            [prefill, decode].iter().all(|values| {
                values.iter().copied().reduce(f64::max).unwrap_or(0.0)
                    <= values.iter().copied().reduce(f64::min).unwrap_or(0.0) * 1.15
            })
        })
}
