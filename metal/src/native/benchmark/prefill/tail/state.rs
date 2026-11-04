use super::*;
use crate::config::PrefillMaterialization::{OutputAndState, StateOnly};

#[test]
#[ignore = "C1 state-only intermediate prefill evaluation, fixed numerical and timing comparison"]
fn compares_prefill_materialization() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let context = positive_env("MIRMIR_BENCH_CONTEXT", 2065)?;
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
    for (block, order) in [
        [OutputAndState, StateOnly, StateOnly, OutputAndState],
        [OutputAndState, StateOnly, StateOnly, OutputAndState],
        [StateOnly, OutputAndState, OutputAndState, StateOnly],
    ]
    .into_iter()
    .enumerate()
    {
        for plan in order {
            model.stream.set_prefill_materialization(plan);
            let observation = run(&mut model, &prompt, checkpoint, config.decode_tokens)?;
            let agrees = reference.as_ref().is_none_or(|tokens| tokens == &observation.tokens);
            writeln!(
                std::io::stderr().lock(),
                "state_only.sample: {}",
                json!({
                    "plan": plan, "block": block, "measured": block > 0,
                    "matches_reference": agrees, "observation": observation,
                })
            )?;
            assert!(agrees, "state-only prefill changed token trajectory");
            if reference.is_none() {
                reference = Some(observation.tokens.clone());
            }
            if block > 0 {
                samples.push((plan, observation.prefill_ms, observation.decode_ms));
                for mode in [OutputAndState, StateOnly] {
                    for values in [
                        samples.iter().filter(|s| s.0 == mode).map(|s| s.1).collect::<Vec<_>>(),
                        samples.iter().filter(|s| s.0 == mode).map(|s| s.2).collect::<Vec<_>>(),
                    ] {
                        if values.len() > 1 {
                            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                            let max = values.iter().copied().fold(0.0, f64::max);
                            assert!(max <= min * 1.15, "unstable state-only comparison");
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
