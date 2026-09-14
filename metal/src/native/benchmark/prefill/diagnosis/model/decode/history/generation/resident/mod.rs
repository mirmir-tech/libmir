use super::*;
mod cohort;
mod readiness;
use cohort::Cohort;

fn epoch() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn stable(values: &[f64], limit: f64) -> bool {
    !values.is_empty()
        && values.iter().all(|v| v.is_finite() && *v > 0.0)
        && values.iter().copied().fold(0.0, f64::max)
            <= values.iter().copied().fold(f64::INFINITY, f64::min) * (1.0 + limit)
}

#[test]
#[ignore = "one bounded resident Qwen C3 matched-context decode reservation experiment"]
fn compares_resident_qwen_generation_reservation() -> Result<()> {
    compare(false)
}

#[test]
#[ignore = "one bounded resident Qwen C3 persistent batch history experiment"]
fn compares_resident_qwen_persistent_history() -> Result<()> {
    compare(true)
}

fn compare(persistent: bool) -> Result<()> {
    use crate::config::HistoryBatching::{Joined, Persistent};
    let mut model = tiles::load_model()?;
    let a = if persistent {
        DecodeReservation::GenerationBudget
    } else {
        DecodeReservation::OnePage
    };
    let b = DecodeReservation::GenerationBudget;
    let mut cohorts = prepare(&mut model, a, b, persistent)?;
    let mut anchors = Vec::new();
    let mut measured = [Vec::new(), Vec::new()];
    let mut blocks = Vec::new();
    for block in 0..6 {
        let order = if block % 2 == 0 {
            [0, 1, 2, 3]
        } else {
            [1, 0, 3, 2]
        };
        let all_positions = cohorts.iter().map(Cohort::positions).collect::<Result<Vec<_>>>()?;
        let positions = all_positions[0].clone();
        assert!(all_positions.iter().all(|p| *p == positions));
        let mut reference = None;
        let mut times = [Vec::new(), Vec::new()];
        for index in order {
            let cohort = &mut cohorts[index];
            let started_epoch_ms = epoch();
            let before = crate::engine::persistent_history::counts();
            let observation = cohort.run(&mut model, block == 0)?;
            let counts: [usize; 2] =
                std::array::from_fn(|i| crate::engine::persistent_history::counts()[i] - before[i]);
            assert_eq!(
                counts,
                if cohort.history == Persistent {
                    if block == 0 {
                        [10, 310]
                    } else {
                        [0, 320]
                    }
                } else {
                    [0, 0]
                }
            );
            let end_epoch_ms = epoch();
            let last = cohort.inputs.iter().map(|input| input.token).collect::<Vec<_>>();
            let matches = reference.as_ref().is_none_or(|(tokens, final_tokens)| {
                *tokens == observation.tokens && *final_tokens == last
            });
            let end_positions = cohort.positions()?;
            assert_eq!(end_positions, positions.iter().map(|p| p + 32).collect::<Vec<_>>());
            writeln!(
                std::io::stderr().lock(),
                "resident.sample: {}",
                json!({
                    "block":block,"cohort":index,"plan":cohort.plan,"history":cohort.history,"builds_appends":counts,"measured":block>=4,
                    "started_epoch_ms":started_epoch_ms,"end_epoch_ms":end_epoch_ms,
                    "start_positions":positions,"end_positions":end_positions,
                    "decode_ms":observation.elapsed_ms,"tokens":observation.tokens,
                    "final_tokens":last,"matches_reference":matches
                })
            )?;
            assert!(matches, "cohorts changed the matched-context token trajectory");
            if reference.is_none() {
                reference = Some((observation.tokens, last));
            }
            let variant = usize::from(cohort.plan != a || cohort.history != Joined);
            times[variant].push(observation.elapsed_ms);
            if (2..4).contains(&block) && variant == 0 {
                anchors.push(observation.elapsed_ms);
            }
            if block >= 4 {
                measured[variant].push(observation.elapsed_ms);
                let mut baseline = anchors.clone();
                baseline.extend(&measured[0]);
                if !stable(&measured[variant], 0.15) || !stable(&baseline, 0.15) {
                    writeln!(
                        std::io::stderr().lock(),
                        "resident.stop: {}",
                        json!({"reason":"measured_stability","block":block,"cohort":index})
                    )?;
                    return Ok(());
                }
            }
        }
        if block == 3 && !stable(&anchors, 0.05) {
            writeln!(
                std::io::stderr().lock(),
                "resident.stop: {}",
                json!({"reason":"baseline_conditioning","samples":anchors})
            )?;
            return Ok(());
        }
        if block >= 4 {
            blocks.push(1.0 - times[1].iter().sum::<f64>() / times[0].iter().sum::<f64>());
        }
    }
    report(&measured, &blocks)
}

fn prepare(
    model: &mut LoadedModel,
    a: DecodeReservation,
    b: DecodeReservation,
    persistent: bool,
) -> Result<Vec<Cohort>> {
    use crate::config::HistoryBatching::Persistent;
    let mut cohorts = [a, b, b, a]
        .into_iter()
        .map(|plan| Cohort::prepare(model, plan))
        .collect::<Result<Vec<_>>>()?;
    if persistent {
        cohorts[1].history = Persistent;
        cohorts[2].history = Persistent;
    }
    assert!(cohorts.iter().all(|c| c.schedule == cohorts[0].schedule));
    model.clear_prefix_cache();
    let memory = crate::engine::memory_stats()?;
    writeln!(
        std::io::stderr().lock(),
        "resident.ready: {}",
        json!({
            "epoch_ms":epoch(), "active_bytes":memory.active,"cached_bytes":memory.cached,
            "peak_bytes_process":memory.peak,"cohorts":4,"generation_budget":256
        })
    )?;
    Ok(cohorts)
}

fn report(measured: &[Vec<f64>; 2], blocks: &[f64]) -> Result<()> {
    assert!(measured.iter().all(|v| v.len() == 4));
    let means = measured.each_ref().map(|v| v.iter().sum::<f64>() / 4.0);
    let reduction = 1.0 - means[1] / means[0];
    writeln!(
        std::io::stderr().lock(),
        "resident.gate: {}",
        json!({
            "baseline_ms":means[0],"candidate_ms":means[1],"reduction":reduction,"blocks":blocks,
            "passes":reduction>=0.03 && blocks.iter().all(|&r|r>0.0)
        })
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
