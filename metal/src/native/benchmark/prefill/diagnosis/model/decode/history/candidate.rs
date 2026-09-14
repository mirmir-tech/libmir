use super::*;
use crate::config::HistoryBatching::{self, Joined, Rows};

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Positions {
    Uniform,
    Ragged,
}

#[test]
#[ignore = "actual Qwen packed history rows numerical gate; set MIRMIR_BENCH_MODEL"]
fn preserves_qwen_history_rows() -> Result<()> {
    numerical(2049, 10)
}

#[test]
#[ignore = "actual GPT-OSS packed history rows numerical gate; set MIRMIR_BENCH_MODEL"]
fn preserves_clamped_history_rows() -> Result<()> {
    numerical(129, 0)
}

fn numerical(context: usize, layers: usize) -> Result<()> {
    let mut model = tiles::load_model()?;
    for positions in [Positions::Uniform, Positions::Ragged] {
        run_modes(&mut model, context, layers, positions, &[Joined, Rows], 3)?;
    }
    Ok(())
}

#[test]
#[ignore = "actual Qwen C3/2049 whole-decode history rows ABBA; set MIRMIR_BENCH_MODEL"]
fn compares_qwen_history_rows() -> Result<()> {
    cost_width(3)
}

#[test]
#[ignore = "Qwen C2 history rows cost; set MIRMIR_BENCH_MODEL"]
fn compares_qwen_history_rows_c2() -> Result<()> {
    cost_width(2)
}

#[test]
#[ignore = "Qwen C5 history rows cost; set MIRMIR_BENCH_MODEL"]
fn compares_qwen_history_rows_c5() -> Result<()> {
    cost_width(5)
}

fn cost_width(width: usize) -> Result<()> {
    let mut model = tiles::load_model()?;
    run_modes(
        &mut model,
        2049,
        10,
        Positions::Uniform,
        &[Joined, Rows, Joined, Rows, Rows, Joined, Rows, Joined, Joined, Rows],
        width,
    )
}

#[test]
#[ignore = "Qwen short/long C2/C5 history rows parity; set MIRMIR_BENCH_MODEL"]
fn preserves_qwen_history_rows_contexts() -> Result<()> {
    let mut model = tiles::load_model()?;
    for width in [2, 5] {
        for context in [257, 8193] {
            run_modes(
                &mut model,
                context,
                if context >= 8192 {
                    0
                } else {
                    10
                },
                Positions::Uniform,
                &[Joined, Rows],
                width,
            )?;
        }
    }
    Ok(())
}

pub(super) fn run_modes(
    model: &mut LoadedModel,
    context: usize,
    layers: usize,
    positions: Positions,
    order: &[HistoryBatching],
    width: usize,
) -> Result<()> {
    let mut reference: Option<super::super::super::Observation> = None;
    let mut samples = [Vec::<f64>::new(), Vec::<f64>::new()];
    for (run, &plan) in order.iter().enumerate() {
        let mut calls = 0;
        let mut gather_calls = 0;
        let mut persistent_calls = [0; 2];
        let observation =
            run_width_with_decode(model, MoePrefill::Default, context, width, |model, inputs| {
                if matches!(positions, Positions::Ragged) {
                    for _ in 0..16 {
                        let outputs = model.decode_batch(&inputs[..2])?;
                        for (input, output) in inputs[..2].iter_mut().zip(outputs) {
                            input.token = greedy_token(&output)?;
                        }
                    }
                }
                model.stream.synchronize()?;
                let before = crate::engine::probe::history::row_batches();
                let before_persistent = crate::engine::persistent_history::counts();
                let before_gather = crate::engine::probe::history::gather_batches();
                model.stream.set_history_batching(plan);
                let result = plain(model, inputs);
                model.stream.set_history_batching(Joined);
                calls = crate::engine::probe::history::row_batches() - before;
                gather_calls = crate::engine::probe::history::gather_batches() - before_gather;
                persistent_calls = std::array::from_fn(|i| {
                    crate::engine::persistent_history::counts()[i] - before_persistent[i]
                });
                result
            })?;
        let matches = reference.as_ref().is_none_or(|prior| {
            prior.tokens == observation.tokens && prior.schedule == observation.schedule
        });
        let memory = crate::engine::memory_stats()?;
        writeln!(
            std::io::stderr().lock(),
            "history.model: {}",
            json!({
                "run": run, "plan": plan, "warmup": run < 2,
                "context": context, "width": width, "positions": positions, "row_batches": calls,
                "gather_batches": gather_calls, "persistent_builds_appends": persistent_calls,
                "matches_reference": matches, "observation": observation,
                "after_decode_memory": {"active": memory.active, "cached": memory.cached, "peak": memory.peak},
            })
        )?;
        assert!(matches, "history batching changed tokens or prefill schedule");
        assert_eq!(
            calls,
            if plan == Rows && matches!(positions, Positions::Uniform) {
                32 * layers
            } else {
                0
            }
        );
        assert_eq!(
            gather_calls,
            if plan == HistoryBatching::Gathered && matches!(positions, Positions::Uniform) {
                32 * layers
            } else {
                0
            }
        );
        assert_eq!(
            persistent_calls.iter().sum::<usize>(),
            if plan == HistoryBatching::Persistent && matches!(positions, Positions::Uniform) {
                32 * layers
            } else {
                0
            }
        );
        if run >= 2 {
            let values = &mut samples[usize::from(plan != Joined)];
            values.push(observation.decode_ms);
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(0.0, f64::max);
            if max > min * 1.15 {
                return Err(crate::engine::Error::BenchmarkStability {
                    case: "history/full-decode".into(),
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
    report(samples, order)
}

fn report(samples: [Vec<f64>; 2], order: &[HistoryBatching]) -> Result<()> {
    if samples.iter().all(|values| values.len() == 4) {
        let blocks = [0, 2].map(|start| {
            1.0 - samples[1][start..start + 2].iter().sum::<f64>()
                / samples[0][start..start + 2].iter().sum::<f64>()
        });
        let means = samples.map(|values| values.iter().sum::<f64>() / 4.0);
        writeln!(
            std::io::stderr().lock(),
            "history.model.gate: {}",
            json!({
                "joined_ms": means[0], "candidate": order[1], "candidate_ms": means[1],
                "reduction": 1.0 - means[1] / means[0], "block_reductions": blocks,
                "passes": means[1] <= means[0] * 0.97 && blocks.iter().all(|&gain| gain > 0.0),
            })
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "remaining C5 short/native-paged history scope; set MIRMIR_BENCH_MODEL"]
fn preserves_qwen_history_rows_contexts_c5() -> Result<()> {
    let mut model = tiles::load_model()?;
    for (context, layers) in [(257, 10), (8193, 0)] {
        run_modes(&mut model, context, layers, Positions::Uniform, &[Joined, Rows], 5)?;
    }
    Ok(())
}
