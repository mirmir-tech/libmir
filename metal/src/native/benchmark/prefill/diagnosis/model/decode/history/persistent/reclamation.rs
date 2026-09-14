use super::*;

#[test]
#[ignore = "actual Qwen batch-history memory reclamation and continuation"]
fn reclaims_qwen_persistent_history() -> Result<()> {
    inspect(2049, 10)
}

#[test]
#[ignore = "actual GPT-OSS history reclamation fallback and continuation"]
fn reclaims_clamped_persistent_history() -> Result<()> {
    inspect(129, 0)
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
    // Inventory handles retain allocations. They are dropped before
    // reclamation.
}

fn inspect(context: usize, layers: usize) -> Result<()> {
    let mut model = tiles::load_model()?;
    model
        .stream
        .set_decode_reservation(crate::config::DecodeReservation::GenerationBudget);
    let mut reference: Option<super::super::super::super::Observation> = None;
    for plan in [Joined, Persistent] {
        let observation = run_budget_with_decode(
            &mut model,
            MoePrefill::Default,
            context,
            3,
            std::num::NonZeroUsize::new(256),
            |model, inputs| {
                model.stream.set_history_batching(plan);
                let initial_counts = crate::engine::persistent_history::counts();
                let mut first = plain(model, inputs)?;
                let before = inventory(model)?;
                let active_before = crate::engine::memory_stats()?.active;
                model.reclaim_decode_reservations()?;
                let active_after = crate::engine::memory_stats()?.active;
                let after = inventory(model)?;
                let second = plain(model, inputs)?;
                let rebuilt = inventory(model)?;
                let counts: [usize; 2] = std::array::from_fn(|i| {
                    crate::engine::persistent_history::counts()[i] - initial_counts[i]
                });
                writeln!(
                    std::io::stderr().lock(),
                    "history.reclaim: {}",
                    json!({
                        "plan":plan,"context":context,"before_allocations_bytes":before,"after_allocations_bytes":after,
                        "rebuilt_allocations_bytes":rebuilt,"active_before":active_before,"active_after":active_after,
                        "builds_appends":counts,
                    })
                )?;
                let selected_layers = if plan == Persistent {
                    layers
                } else {
                    0
                };
                assert_eq!(before.0, 2 * selected_layers);
                assert_eq!(after, (0, 0));
                assert_eq!(rebuilt.0, before.0);
                assert_eq!(counts, [2 * selected_layers, 62 * selected_layers]);
                if selected_layers > 0 {
                    assert_eq!(before.1, 135 * 1024 * 1024);
                    assert!(active_before.saturating_sub(active_after) >= before.1);
                }
                for (tokens, tail) in first.tokens.iter_mut().zip(second.tokens) {
                    tokens.extend(tail);
                }
                first.elapsed_ms += second.elapsed_ms;
                model.stream.set_history_batching(Joined);
                Ok(first)
            },
        )?;
        let matches = reference
            .as_ref()
            .is_none_or(|r| r.tokens == observation.tokens && r.schedule == observation.schedule);
        writeln!(
            std::io::stderr().lock(),
            "history.reclaim.tokens: {}",
            json!({"plan":plan,"matches":matches,"observation":observation})
        )?;
        assert!(matches);
        reference.get_or_insert(observation);
    }
    Ok(())
}
