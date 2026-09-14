use super::*;

#[test]
#[ignore = "one bounded Joined-only Qwen readiness probe; requires external GPU telemetry gate"]
fn checks_resident_qwen_decode_readiness() -> Result<()> {
    let mut model = tiles::load_model()?;
    let mut cohorts = (0..4)
        .map(|_| Cohort::prepare_budget(&mut model, DecodeReservation::GenerationBudget, 3, 512))
        .collect::<Result<Vec<_>>>()?;
    assert!(cohorts.iter().all(|c| c.schedule == cohorts[0].schedule));
    model.clear_prefix_cache();
    let memory = crate::engine::memory_stats()?;
    writeln!(
        std::io::stderr().lock(),
        "readiness.ready: {}",
        json!({"epoch_ms":epoch(),"active_bytes":memory.active,"cached_bytes":memory.cached,
            "cohorts":4,"width":3,"generation_budget":512,"history":"joined"})
    )?;
    let mut measured = Vec::new();
    for round in 0..3 {
        let positions = cohorts[0].positions()?;
        assert!(cohorts.iter().all(|c| c.positions().is_ok_and(|p| p == positions)));
        let mut reference: Option<(Vec<Vec<u32>>, Vec<u32>)> = None;
        let order = if round % 2 == 0 {
            [0, 1, 2, 3]
        } else {
            [3, 2, 1, 0]
        };
        for index in order {
            let cohort = &mut cohorts[index];
            let started_epoch_ms = epoch();
            let started = Instant::now();
            let mut tokens = vec![Vec::new(); 3];
            let mut decode_ms = 0.0;
            for _ in 0..4 {
                let observation = cohort.run(&mut model, false)?;
                decode_ms += observation.elapsed_ms;
                for (row, part) in tokens.iter_mut().zip(observation.tokens) {
                    row.extend(part);
                }
            }
            let window_ms = started.elapsed().as_secs_f64() * 1000.0;
            let end_epoch_ms = epoch();
            let final_tokens = cohort.inputs.iter().map(|input| input.token).collect::<Vec<_>>();
            let end_positions = cohort.positions()?;
            assert_eq!(end_positions, positions.iter().map(|p| p + 128).collect::<Vec<_>>());
            let matches = reference
                .as_ref()
                .is_none_or(|(expected, last)| *expected == tokens && *last == final_tokens);
            // Own the reference; every cohort follows an independent recurrent trajectory.
            writeln!(
                std::io::stderr().lock(),
                "readiness.window: {}",
                json!({"round":round,"cohort":index,"measured":round>0,
                    "started_epoch_ms":started_epoch_ms,"end_epoch_ms":end_epoch_ms,
                    "decode_ms":decode_ms,"window_ms":window_ms,"start_positions":positions,
                    "end_positions":end_positions,"tokens":tokens,"final_tokens":final_tokens,
                    "matches_reference":matches})
            )?;
            assert!(matches, "baseline cohorts must preserve the matched-context trajectory");
            if reference.is_none() {
                reference = Some((tokens, final_tokens));
            }
            if round > 0 {
                measured.push(decode_ms);
            }
        }
    }
    writeln!(
        std::io::stderr().lock(),
        "readiness.decode_gate: {}",
        json!({"samples":measured,"passes":stable(&measured,0.05),
            "requires_external_telemetry_gate":true})
    )?;
    Ok(())
}
