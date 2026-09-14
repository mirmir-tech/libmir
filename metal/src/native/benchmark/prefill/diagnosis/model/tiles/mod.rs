mod settled;

use super::*;

#[test]
#[ignore = "whole-model compact BN64 MoE C5/2k/8k parity; set MIRMIR_BENCH_MODEL"]
fn preserves_compact_tiles_model() -> Result<()> {
    let mut model = load_model()?;
    for context in [2049, 8193] {
        let reference = run(&mut model, MoePrefill::Default, context)?;
        writeln!(
            std::io::stderr().lock(),
            "moe.tiles_model: {}",
            json!({
                "context": context, "plan": "grouped_fused", "observation": reference,
            })
        )?;
        let candidate = run(&mut model, MoePrefill::Tiles64, context)?;
        writeln!(
            std::io::stderr().lock(),
            "moe.tiles_model: {}",
            json!({
                "context": context, "plan": "grouped_tiles64", "observation": candidate,
            })
        )?;
        assert_eq!(reference.tokens, candidate.tokens, "full-model greedy parity");
        assert_eq!(reference.schedule, candidate.schedule, "full-model scheduling parity");
    }
    Ok(())
}

#[test]
#[ignore = "warmed full-model BN64 C5/2049 timing gate; set MIRMIR_BENCH_MODEL"]
fn compares_compact_tiles_model() -> Result<()> {
    let mut model = load_model()?;
    let mut reference: Option<Observation> = None;
    let mut samples = Vec::new();
    for block in 0..4 {
        let order = if block % 2 == 0 {
            [MoePrefill::Default, MoePrefill::Tiles64, MoePrefill::Tiles64, MoePrefill::Default]
        } else {
            [MoePrefill::Tiles64, MoePrefill::Default, MoePrefill::Default, MoePrefill::Tiles64]
        };
        // One warm run of each plan; measured blocks alternate ABBA/BAAB.
        let order = if block == 0 {
            &order[..2]
        } else {
            &order[..]
        };
        for (slot, &mode) in order.iter().enumerate() {
            let observation = run(&mut model, mode, 2049)?;
            let matches = reference.as_ref().is_none_or(|prior| {
                prior.tokens == observation.tokens && prior.schedule == observation.schedule
            });
            writeln!(
                std::io::stderr().lock(),
                "moe.tiles_timing: {}",
                json!({
                    "block": block, "slot": slot, "measured": block > 0,
                    "plan": name(mode), "matches_reference": matches, "observation": observation,
                })
            )?;
            assert!(matches, "full-model tokens or schedule changed");
            if block > 0 {
                samples.push((mode, observation.prefill_ms, observation.decode_ms));
                assert!(
                    stable(&samples),
                    "unstable prefill/decode control; abort timing qualification"
                );
            }
            if reference.is_none() {
                reference = Some(observation);
            }
        }
    }
    Ok(())
}

pub(super) fn load_model() -> Result<LoadedModel> {
    initialize_tracing();
    let mut config = BenchmarkConfig::from_env()?;
    config.prompt_tokens = 8193;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.set_max_batch_requests(5);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    model.info.prefill_step = 512;
    Ok(model)
}
