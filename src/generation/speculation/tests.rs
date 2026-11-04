use super::*;

#[test]
fn speculative_acceptance_cannot_exceed_the_remaining_completion_budget() -> Result<()> {
    let settings = models::generation::GenerationConfig::default()
        .resolve(models::generation::GenerationOverrides::default())?;
    let cycle =
        crate::generation::CycleRecovery::new(settings, None, 256, &SamplingLogits::None, None)?;
    for remaining in 1..=4 {
        let bounds = draft_bounds(None, &cycle, &SamplingLogits::None, &[255], &[], remaining)?
            .ok_or_else(|| crate::Error::InvalidResponseFormat("no speculative bounds".into()))?;
        let (accepted, next) = bounds.apply(&[1, 2, 3], 3, 4);
        assert_eq!(accepted + 1, remaining);
        assert_eq!(next as usize, remaining);
    }
    Ok(())
}
