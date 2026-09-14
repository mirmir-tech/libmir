use super::*;
use crate::{
    config::GdnExecution::{self, Native, PackedDecode, PackedPrefill},
    engine::diagnostics::gdn_experiment_calls,
};

mod quality;

#[test]
#[ignore = "Qwen GDN four-lane full-model parity; set MIRMIR_BENCH_MODEL"]
fn preserves_four_lane_gdn_model() -> Result<()> {
    let mut model = tiles::load_model()?;
    for (width, context) in [(3, 2049), (5, 257)] {
        run_modes(&mut model, width, context, &[Native, PackedDecode, PackedPrefill])?;
    }
    quality::check(&mut model)
}

#[test]
#[ignore = "Qwen four-lane GDN decode cost gate; set MIRMIR_BENCH_MODEL"]
fn compares_four_lane_gdn_decode() -> Result<()> {
    cost(PackedDecode)
}

#[test]
#[ignore = "Qwen four-lane GDN prefill cost gate; set MIRMIR_BENCH_MODEL"]
fn compares_four_lane_gdn_prefill() -> Result<()> {
    cost(PackedPrefill)
}

fn cost(candidate: GdnExecution) -> Result<()> {
    let mut model = tiles::load_model()?;
    run_modes(
        &mut model,
        3,
        2049,
        &[
            Native, candidate, Native, candidate, candidate, Native, candidate, Native, Native,
            candidate,
        ],
    )
}

fn run_modes(
    model: &mut LoadedModel,
    width: usize,
    context: usize,
    order: &[GdnExecution],
) -> Result<()> {
    let mut reference: Option<super::super::Observation> = None;
    let mut samples = [Vec::<f64>::new(), Vec::<f64>::new()];
    for (run, &plan) in order.iter().enumerate() {
        model.stream.set_gdn_execution(if plan == PackedPrefill {
            plan
        } else {
            Native
        });
        let before = gdn_experiment_calls();
        let mut decode_calls = 0;
        let observation =
            run_width_with_decode(model, MoePrefill::Default, context, width, |model, inputs| {
                model.stream.set_gdn_execution(if plan == PackedDecode {
                    plan
                } else {
                    Native
                });
                let before = gdn_experiment_calls();
                let result = plain(model, inputs);
                decode_calls = gdn_experiment_calls() - before;
                model.stream.set_gdn_execution(Native);
                result
            })?;
        let calls = gdn_experiment_calls() - before;
        let matches = reference.as_ref().is_none_or(|prior| {
            prior.tokens == observation.tokens && prior.schedule == observation.schedule
        });
        writeln!(
            std::io::stderr().lock(),
            "gdn.model: {}",
            json!({
                "run":run,"plan":plan,"warmup":run<2,"width":width,"context":context,
                "calls":calls,"decode_calls":decode_calls,"matches_reference":matches,
                "observation":observation,
            })
        )?;
        assert!(matches, "GDN candidate changed model tokens or prefill schedule");
        match plan {
            Native => assert_eq!(calls, 0),
            PackedDecode => assert_eq!(decode_calls, 32 * 30),
            PackedPrefill => {
                assert!(calls > 0);
                assert_eq!(decode_calls, 0);
            },
        }
        if order.len() == 10 && run >= 2 {
            let elapsed = if order.contains(&PackedPrefill) {
                observation.prefill_ms
            } else {
                observation.decode_ms
            };
            let values = &mut samples[usize::from(plan != Native)];
            values.push(elapsed);
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(0.0, f64::max);
            if max > min * 1.15 {
                return Err(crate::engine::Error::BenchmarkStability {
                    case: "GDN/full-model".into(),
                    variant: format!("{plan:?}"),
                    spread_percent: (max / min - 1.0) * 100.0,
                }
                .into());
            }
        }
        if reference.is_none() {
            reference = Some(observation);
        }
    }
    if samples.iter().all(|v| v.len() == 4) {
        let blocks = [0, 2].map(|start| {
            1.0 - samples[1][start..start + 2].iter().sum::<f64>()
                / samples[0][start..start + 2].iter().sum::<f64>()
        });
        let means = samples.map(|v| v.iter().sum::<f64>() / 4.0);
        writeln!(
            std::io::stderr().lock(),
            "gdn.model.gate: {}",
            json!({
                "candidate":order[1],"native_ms":means[0],"candidate_ms":means[1],
                "reduction":1.0-means[1]/means[0],"block_reductions":blocks,
                "passes":means[1]<=means[0]*0.97 && blocks.iter().all(|&v|v>0.0),
            })
        )?;
    }
    Ok(())
}
