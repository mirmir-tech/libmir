use super::*;

#[test]
#[ignore = "Qwen C3/2049 decode reservation whole-model ABBA; set MIRMIR_BENCH_MODEL"]
fn compares_qwen_decode_reservation() -> Result<()> {
    let mut model = tiles::load_model()?;
    let mut reference = None;
    let mut samples = [Vec::<[f64; 2]>::new(), Vec::new()];
    let a = DecodeReservation::OnePage;
    let b = horizon()?;
    for (run, plan) in [a, b, a, b, b, a, b, a, a, b].into_iter().enumerate() {
        model.stream.set_decode_reservation(plan);
        let result = run_width_with_decode(&mut model, MoePrefill::Default, 2049, 3, plain);
        model.stream.set_decode_reservation(a);
        let observation = result?;
        let matches =
            reference.as_ref().is_none_or(|r: &super::super::super::super::Observation| {
                r.tokens == observation.tokens && r.schedule == observation.schedule
            });
        writeln!(
            std::io::stderr().lock(),
            "reservation.model: {}",
            json!({
                "run":run,"plan":plan,"warmup":run<2,"matches_reference":matches,"observation":observation,
            })
        )?;
        assert!(matches);
        if run >= 2 {
            let values = &mut samples[usize::from(plan != a)];
            values.push([observation.prefill_ms, observation.decode_ms]);
            for metric in 0..2 {
                let min = values.iter().map(|v| v[metric]).fold(f64::INFINITY, f64::min);
                let max = values.iter().map(|v| v[metric]).fold(0.0, f64::max);
                if max > min * 1.15 {
                    return Err(crate::engine::Error::BenchmarkStability {
                        case: format!(
                            "reservation/{}",
                            if metric == 0 {
                                "prefill"
                            } else {
                                "decode"
                            }
                        ),
                        variant: format!("{plan:?}"),
                        spread_percent: (max / min - 1.0) * 100.0,
                    }
                    .into());
                }
            }
        }
        if reference.is_none() {
            reference = Some(observation);
        }
    }
    for (metric, name) in [(0, "prefill"), (1, "decode")] {
        let means = samples.each_ref().map(|v| v.iter().map(|s| s[metric]).sum::<f64>() / 4.0);
        let blocks = [0, 2].map(|i| {
            1.0 - samples[1][i..i + 2].iter().map(|s| s[metric]).sum::<f64>()
                / samples[0][i..i + 2].iter().map(|s| s[metric]).sum::<f64>()
        });
        let reduction = 1.0 - means[1] / means[0];
        let passes = if metric == 0 {
            reduction >= -0.01 && blocks.iter().all(|&b| b >= -0.01)
        } else {
            reduction >= 0.03 && blocks.iter().all(|&b| b > 0.0)
        };
        writeln!(
            std::io::stderr().lock(),
            "reservation.gate: {}",
            json!({"metric":name,"baseline_ms":means[0],"candidate_ms":means[1],"reduction":reduction,"blocks":blocks,"passes":passes})
        )?;
    }
    Ok(())
}
