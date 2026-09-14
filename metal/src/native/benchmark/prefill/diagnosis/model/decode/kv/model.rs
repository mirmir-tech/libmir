use super::*;
use crate::config::KeyValueProjection::{self, JoinedDecode, Separate};

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Positions {
    Uniform,
    Ragged,
}

#[test]
#[ignore = "actual Qwen packed K/V decode join numerical gate; set MIRMIR_BENCH_MODEL"]
fn preserves_qwen_key_value_decode() -> Result<()> {
    numerical(2049, 10)
}

#[test]
#[ignore = "actual GPT-OSS packed K/V decode join numerical gate; set MIRMIR_BENCH_MODEL"]
fn preserves_clamped_key_value_decode() -> Result<()> {
    numerical(129, 24)
}

fn numerical(context: usize, layers: usize) -> Result<()> {
    let mut model = tiles::load_model()?;
    for positions in [Positions::Uniform, Positions::Ragged] {
        run_modes(&mut model, context, layers, positions, &[Separate, JoinedDecode])?;
    }
    Ok(())
}

#[test]
#[ignore = "actual Qwen C3/2049 whole-decode K/V join ABBA; set MIRMIR_BENCH_MODEL"]
fn compares_qwen_key_value_decode() -> Result<()> {
    let mut model = tiles::load_model()?;
    run_modes(
        &mut model,
        2049,
        10,
        Positions::Uniform,
        &[
            Separate, JoinedDecode, Separate, JoinedDecode, JoinedDecode, Separate, JoinedDecode,
            Separate, Separate, JoinedDecode,
        ],
    )
}

fn run_modes(
    model: &mut LoadedModel,
    context: usize,
    layers: usize,
    positions: Positions,
    order: &[KeyValueProjection],
) -> Result<()> {
    let mut reference: Option<super::super::super::Observation> = None;
    let mut samples = [Vec::<f64>::new(), Vec::<f64>::new()];
    for (run, &plan) in order.iter().enumerate() {
        let mut calls = 0;
        let observation =
            run_width_with_decode(model, MoePrefill::Default, context, 3, |model, inputs| {
                if matches!(positions, Positions::Ragged) {
                    for _ in 0..16 {
                        let outputs = model.decode_batch(&inputs[..2])?;
                        for (input, output) in inputs[..2].iter_mut().zip(outputs) {
                            input.token = greedy_token(&output)?;
                        }
                    }
                }
                model.stream.synchronize()?;
                let before = crate::engine::probe::joined_calls();
                model.stream.set_key_value_projection(plan);
                let result = plain(model, inputs);
                model.stream.set_key_value_projection(Separate);
                calls = crate::engine::probe::joined_calls() - before;
                result
            })?;
        let matches = reference.as_ref().is_none_or(|prior| {
            prior.tokens == observation.tokens && prior.schedule == observation.schedule
        });
        let memory = crate::engine::memory_stats()?;
        writeln!(
            std::io::stderr().lock(),
            "kv.model: {}",
            json!({
                "run": run, "plan": plan, "warmup": run < 2,
                "context": context, "positions": positions, "joined_calls": calls,
                "matches_reference": matches, "observation": observation,
                "after_decode_memory": {"active": memory.active, "cached": memory.cached, "peak": memory.peak},
            })
        )?;
        assert!(matches, "K/V decode join changed tokens or prefill schedule");
        assert_eq!(
            calls,
            if plan == JoinedDecode {
                32 * layers
            } else {
                0
            }
        );
        if run >= 2 {
            let values = &mut samples[usize::from(plan != Separate)];
            values.push(observation.decode_ms);
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(0.0, f64::max);
            if max > min * 1.15 {
                return Err(crate::engine::Error::BenchmarkStability {
                    case: "KV/full-decode".into(),
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
    if samples.iter().all(|values| values.len() == 4) {
        let blocks = [0, 2].map(|start| {
            1.0 - samples[1][start..start + 2].iter().sum::<f64>()
                / samples[0][start..start + 2].iter().sum::<f64>()
        });
        let means = samples.map(|values| values.iter().sum::<f64>() / 4.0);
        writeln!(
            std::io::stderr().lock(),
            "kv.model.gate: {}",
            json!({
                "separate_ms": means[0], "joined_ms": means[1],
                "reduction": 1.0 - means[1] / means[0], "block_reductions": blocks,
                "passes": means[1] <= means[0] * 0.97 && blocks.iter().all(|&gain| gain > 0.0),
            })
        )?;
    }
    Ok(())
}
