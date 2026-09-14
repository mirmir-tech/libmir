use super::*;

fn epoch() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[test]
#[ignore = "one baseline-only prefill drift diagnostic with an idle recovery sample; set MIRMIR_BENCH_MODEL"]
fn diagnoses_prefill_drift_with_recovery() -> Result<()> {
    let mut model = tiles::load_model()?;
    let mut reference: Option<Observation> = None;
    for run in 0..7 {
        if run == 6 {
            writeln!(
                std::io::stderr().lock(),
                "prefill.drift_idle: {}",
                json!({"epoch_ms":epoch(),"duration_ms":20000})
            )?;
            std::thread::sleep(std::time::Duration::from_secs(20));
        }
        writeln!(
            std::io::stderr().lock(),
            "prefill.drift_start: {}",
            json!({"run":run,"epoch_ms":epoch()})
        )?;
        let observation = run_budget_with_decode(
            &mut model,
            MoePrefill::Default,
            2049,
            3,
            std::num::NonZeroUsize::new(32),
            decode::plain,
        )?;
        let matches = reference.as_ref().is_none_or(|prior| {
            prior.tokens == observation.tokens && prior.schedule == observation.schedule
        });
        writeln!(
            std::io::stderr().lock(),
            "prefill.drift_end: {}",
            json!({
                "run":run,"epoch_ms":epoch(),"matches_reference":matches,"observation":observation
            })
        )?;
        assert!(matches, "diagnostic changed tokens or execution schedule");
        if reference.is_none() {
            reference = Some(observation);
        }
    }
    Ok(())
}
