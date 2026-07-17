use super::*;

fn trace(values: &[f32]) -> LogitsTrace {
    LogitsTrace {
        shape: vec![1, 1, i32::try_from(values.len()).unwrap_or(0)],
        values: values.to_vec(),
    }
}

fn candidates(token_ids: &[u32], scores: &[f32]) -> CandidateLogitsTrace {
    CandidateLogitsTrace {
        token_ids: token_ids.to_vec(),
        scores: scores.to_vec(),
    }
}

#[test]
fn greedy_sampling_picks_highest_logit() -> Result<()> {
    let mut sampler = Sampler::new(SamplerConfig::default())?;
    let token = sampler.sample(&trace(&[-1.0, 4.0, 3.0]))?;

    assert_eq!(token, 1);
    Ok(())
}

#[test]
fn top_k_one_forces_highest_logit() -> Result<()> {
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 1.0,
        top_k: 1,
        ..SamplerConfig::default()
    })?;
    let token = sampler.sample(&trace(&[2.0, 9.0, 8.0]))?;

    assert_eq!(token, 1);
    Ok(())
}

#[test]
fn candidate_sampling_uses_candidate_token_ids() -> Result<()> {
    let mut sampler = Sampler::new(SamplerConfig::default())?;
    let token =
        sampler.sample_candidates_with_history(&candidates(&[42, 7, 99], &[1.0, 5.0, 2.0]), &[])?;

    assert_eq!(token, 7);
    Ok(())
}

#[test]
fn rejects_invalid_top_p() {
    let config = SamplerConfig { top_p: 0.0, ..SamplerConfig::default() };

    assert!(Sampler::new(config).is_err());
}

#[test]
fn repetition_penalty_demotes_seen_token() -> Result<()> {
    let mut sampler = Sampler::new(SamplerConfig {
        repetition_penalty: 2.0,
        ..SamplerConfig::default()
    })?;
    let token = sampler.sample_with_history(&trace(&[0.0, 6.0, 4.0]), &[1])?;

    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn top_p_is_applied_before_top_k() -> Result<()> {
    let sampler = Sampler::new(SamplerConfig {
        temperature: 1.0,
        top_p: 0.69,
        top_k: 2,
        ..SamplerConfig::default()
    })?;
    let candidates = &mut candidate::candidates(
        &[0.5_f32.ln(), 0.2_f32.ln(), 0.2_f32.ln(), 0.1_f32.ln()],
        &[],
        1.0,
    )?;
    candidates.sort_by(|left, right| right.score.total_cmp(&left.score));

    let weights = sampler.filtered_weights(candidates);

    assert_eq!(
        weights.iter().map(|candidate| candidate.token_id).collect::<Vec<_>>(),
        vec![0, 1]
    );
    Ok(())
}
