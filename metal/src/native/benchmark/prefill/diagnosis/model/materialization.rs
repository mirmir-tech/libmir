use super::*;
use crate::config::PrefillMaterialization::{OutputAndState, StateOnly};

#[test]
#[ignore = "actual-model state-only C3/C5, exact-prefix and checkpoint continuation qualification"]
fn qualifies_state_only_prefill() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 16;
    metal.set_max_batch_requests(5);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    model.info.prefill_step = 512;
    for width in [3, 5] {
        let mut reference: Option<Observation> = None;
        for mode in [OutputAndState, StateOnly] {
            model.stream.set_prefill_materialization(mode);
            let observation =
                run_width_with_decode(&mut model, MoePrefill::Default, 2049, width, decode::plain)?;
            if let Some(reference) = &reference {
                assert_eq!(reference.tokens, observation.tokens);
                assert_eq!(reference.schedule, observation.schedule);
            }
            writeln!(
                std::io::stderr().lock(),
                "state_only.batch: {}",
                json!({
                    "mode": mode, "width": width, "observation": observation,
                })
            )?;
            reference = Some(observation);
        }
    }
    let mut reference = None;
    for mode in [OutputAndState, StateOnly] {
        model.stream.set_prefill_materialization(mode);
        model.clear_prefix_cache();
        let mut trajectories = Vec::new();
        let mut prompt = (1_000..3_065).collect::<Vec<u32>>();
        for expected_reuse in [0, 2065, 2048] {
            if expected_reuse == 2048 {
                prompt[2060] = 42;
            }
            let session = Uuid::new_v4();
            let output = model.prefill(
                session,
                &prompt,
                &[2048],
                &SamplingLogits::None,
                Some(16),
                &mut |_| {},
            )?;
            assert_eq!(output.prefix_cache_tokens, expected_reuse);
            let mut token = greedy_token(&output.output)?;
            let mut tokens = vec![token];
            for _ in 1..32 {
                token = greedy_token(&model.decode(session, token, SamplingLogits::None)?)?;
                tokens.push(token);
            }
            model.release_session(session)?;
            trajectories.push(tokens);
        }
        assert_eq!(trajectories[0], trajectories[1]);
        if let Some(reference) = &reference {
            assert_eq!(&trajectories, reference);
        }
        writeln!(
            std::io::stderr().lock(),
            "state_only.prefix: {}",
            json!({"mode": mode, "tokens": trajectories})
        )?;
        reference = Some(trajectories);
    }
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(())
}
