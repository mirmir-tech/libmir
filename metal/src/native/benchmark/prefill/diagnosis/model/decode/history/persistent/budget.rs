use super::*;
use crate::engine::{MemoryStats, persistent_history::budget::Admission};

#[test]
#[ignore = "actual Qwen quota fallback and simulated pressure recovery; no timing"]
fn bounds_qwen_history_and_recovers_from_pressure() -> Result<()> {
    inspect(2049, 10)
}

#[test]
#[ignore = "actual GPT-OSS quota/pressure fallback protection; no timing"]
fn bounds_clamped_history_and_recovers_from_pressure() -> Result<()> {
    inspect(129, 0)
}

fn pressure(active: usize) -> MemoryStats {
    MemoryStats {
        active,
        cached: 0,
        peak: 0,
        limit: 100,
        recommended: Some(100),
    }
}

fn inventory(model: &LoadedModel) -> Result<(usize, usize)> {
    let mut allocations = std::collections::HashSet::new();
    for state in model.sessions.values() {
        allocations.extend(state.cache.history_allocations()?);
    }
    Ok((
        allocations.len(),
        allocations.iter().map(mirtal::memory::Allocation::bytes).sum(),
    ))
}

fn inspect(context: usize, layers: usize) -> Result<()> {
    let mut model = tiles::load_model()?;
    model
        .stream
        .set_decode_reservation(crate::config::DecodeReservation::GenerationBudget);
    let layer_bytes = 135 * 1024 * 1024 / 10;
    let mut reference: Option<super::super::super::super::Observation> = None;
    for (plan, limit, expected_layers) in [
        (Joined, 512 * 1024 * 1024, 0),
        (Persistent, 0, 0),
        (Persistent, 3 * layer_bytes, layers.min(3)),
        (Persistent, 512 * 1024 * 1024, layers),
    ] {
        model.stream.history_budget().set_limit(limit);
        let observation = run_budget_with_decode(
            &mut model,
            MoePrefill::Default,
            context,
            3,
            std::num::NonZeroUsize::new(256),
            |model, inputs| {
                model.stream.set_history_batching(plan);
                model.apply_history_pressure(pressure(40))?;
                let before = crate::engine::persistent_history::counts();
                let mut result = plain(model, inputs)?;
                let initial = inventory(model)?;
                assert_eq!(initial, (2 * expected_layers, expected_layers * layer_bytes));
                assert_eq!(model.stream.history_budget().snapshot().retained_bytes, initial.1);
                let prefixes = model.prefixes.group_count();
                model.apply_history_pressure(pressure(60))?;
                assert_eq!(model.prefixes.group_count(), prefixes);
                assert_eq!(
                    model.stream.history_budget().snapshot().admission,
                    Admission::Suspended
                );
                assert_eq!(inventory(model)?, (0, 0));
                assert_eq!(model.stream.history_budget().snapshot().retained_bytes, 0);
                let blocked = plain(model, inputs)?;
                assert_eq!(inventory(model)?, (0, 0));
                model.apply_history_pressure(pressure(40))?;
                let resumed = plain(model, inputs)?;
                let restored = inventory(model)?;
                assert_eq!(restored, initial);
                let calls: [usize; 2] = std::array::from_fn(|i| {
                    crate::engine::persistent_history::counts()[i] - before[i]
                });
                assert_eq!(calls, [2 * expected_layers, 62 * expected_layers]);
                writeln!(
                    std::io::stderr().lock(),
                    "history.budget: {}",
                    json!({
                        "plan":plan,"limit_bytes":limit,"context":context,"initial_allocations_bytes":initial,
                        "restored_allocations_bytes":restored,"prefix_groups_preserved":prefixes,
                        "builds_appends":calls,"simulated_pressure":true,"budget":model.stream.history_budget().snapshot(),
                    })
                )?;
                for tail in [blocked, resumed] {
                    for (tokens, more) in result.tokens.iter_mut().zip(tail.tokens) {
                        tokens.extend(more);
                    }
                    result.elapsed_ms += tail.elapsed_ms;
                }
                model.stream.set_history_batching(Joined);
                Ok(result)
            },
        )?;
        let matches = reference
            .as_ref()
            .is_none_or(|r| r.tokens == observation.tokens && r.schedule == observation.schedule);
        writeln!(
            std::io::stderr().lock(),
            "history.budget.tokens: {}",
            json!({"plan":plan,"limit_bytes":limit,"matches":matches,"observation":observation})
        )?;
        assert!(matches);
        assert_eq!(model.stream.history_budget().snapshot().retained_bytes, 0);
        reference.get_or_insert(observation);
    }
    Ok(())
}
