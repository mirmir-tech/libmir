use super::*;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Warmup,
    Control,
    Components,
}

#[test]
#[ignore = "bounded matched-context C2/C3 component diagnosis; set MIRMIR_BENCH_MODEL"]
fn diagnoses_packed_width_components() -> Result<()> {
    let mut model = tiles::load_model()?;
    let mut references: [Option<super::super::Observation>; 2] = [None, None];
    let mut controls = [Vec::<f64>::new(), Vec::<f64>::new()];
    // One warmup per width; controls bracket reverse-order sparse profiling.
    // No candidate, adaptive warmup, discarded outliers or reference-engine runs.
    let runs = [
        (2, Phase::Warmup),
        (3, Phase::Warmup),
        (2, Phase::Control),
        (3, Phase::Control),
        (3, Phase::Components),
        (2, Phase::Components),
        (3, Phase::Control),
        (2, Phase::Control),
    ];
    for (run, (width, phase)) in runs.into_iter().enumerate() {
        let span = tracing::debug_span!("decode_width", width, ?phase).entered();
        let mut boundaries = None;
        let observation = run_width_with_decode(
            &mut model,
            MoePrefill::Default,
            2049,
            width,
            |model, inputs| match phase {
                Phase::Components => components::measure(model, inputs),
                Phase::Warmup | Phase::Control => {
                    let (observation, sample) = probe::measure(model, inputs)?;
                    boundaries = Some(sample);
                    Ok(observation)
                },
            },
        )?;
        drop(span);
        let reference = &mut references[width - 2];
        let matches = reference.as_ref().is_none_or(|prior| {
            prior.tokens == observation.tokens && prior.schedule == observation.schedule
        });
        writeln!(
            std::io::stderr().lock(),
            "decode.width: {}",
            json!({
                "run": run, "width": width, "phase": phase, "context": 2049,
                "matches_reference": matches, "observation": observation,
                "boundaries": boundaries,
            })
        )?;
        assert!(matches, "same-width tokens or prefill schedule changed");
        if matches!(phase, Phase::Control) {
            let samples = &mut controls[width - 2];
            samples.push(observation.decode_ms);
            let min = samples.iter().copied().reduce(f64::min).unwrap_or(0.0);
            let max = samples.iter().copied().reduce(f64::max).unwrap_or(0.0);
            assert!(max <= min * 1.15, "decode control spread exceeds 15%; stop diagnosis");
        }
        if reference.is_none() {
            *reference = Some(observation);
        }
    }
    Ok(())
}
