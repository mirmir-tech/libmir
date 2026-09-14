use super::*;

#[test]
#[ignore = "Qwen budget256 page ownership/reclaim diagnostic; no timing qualification"]
fn inspects_qwen_budget256_resource_cost() -> Result<()> {
    let mut model = tiles::load_model()?;
    let mut reference = None;
    for plan in [DecodeReservation::OnePage, DecodeReservation::GenerationBudget] {
        assert_eq!(model.stream.paged_arenas().resident_arenas()?, 0);
        model.stream.set_decode_reservation(plan);
        let observation = run_budget_with_decode(
            &mut model,
            MoePrefill::Default,
            2049,
            3,
            std::num::NonZeroUsize::new(256),
            |model, inputs| {
                let before = model.stream.paged_arenas().resident_metrics()?;
                let headroom_before = model.prefill_page_headroom()?;
                model.reclaim_decode_reservations()?;
                let after = model.stream.paged_arenas().resident_metrics()?;
                assert_eq!(before.len(), 10);
                assert_eq!(after.len(), before.len());
                for (before, after) in before.iter().zip(&after) {
                    assert_eq!(before.capacity_pages, after.capacity_pages);
                    assert_eq!(before.buffer_bytes, after.buffer_bytes);
                    let expected = if plan == DecodeReservation::GenerationBudget {
                        45
                    } else {
                        0
                    };
                    assert_eq!(before.owned_pages - after.owned_pages, expected);
                }
                writeln!(
                    std::io::stderr().lock(),
                    "budget256.resources: {}",
                    json!({
                        "plan":plan,"before":before,"after":after,
                        "headroom_before":headroom_before,"headroom_after":model.prefill_page_headroom()?
                    })
                )?;
                let mut tokens = inputs.iter().map(|i| vec![i.token]).collect::<Vec<_>>();
                for _ in 0..4 {
                    let outputs = model.decode_batch(inputs)?;
                    for ((input, output), tokens) in inputs.iter_mut().zip(outputs).zip(&mut tokens)
                    {
                        input.token = greedy_token(&output)?;
                        tokens.push(input.token);
                    }
                }
                model.stream.synchronize()?;
                Ok(Observation { tokens, elapsed_ms: 0.0 })
            },
        )?;
        let matches =
            reference
                .as_ref()
                .is_none_or(|prior: &super::super::super::super::Observation| {
                    prior.tokens == observation.tokens && prior.schedule == observation.schedule
                });
        writeln!(
            std::io::stderr().lock(),
            "budget256.prefill: {}",
            json!({
                "plan":plan,"matches_reference":matches,"observation":observation,
                "timing_is_diagnostic_only":true
            })
        )?;
        assert!(matches);
        model.clear_prefix_cache();
        assert_eq!(model.stream.paged_arenas().resident_arenas()?, 0);
        if reference.is_none() {
            reference = Some(observation);
        }
    }
    Ok(())
}
