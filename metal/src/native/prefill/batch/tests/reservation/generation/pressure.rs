use super::*;

#[test]
fn unfinished_prefill_returns_optional_pages_for_another_request() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = bounded(family)?;
        let mut first = request(&model, 0, usize::MAX);
        first.prompt_tokens = (0..64).collect();
        let reference_id = Uuid::new_v4();
        let reference = model.prefill(
            reference_id,
            &first.prompt_tokens,
            &[],
            SamplingLogits::Full,
            None,
            &mut |_| {},
        )?;
        let expected = admission::token(&model, reference.output)?;
        model.release_session(reference_id)?;
        model.clear_prefix_cache();
        let (batch, _) =
            MetalPrefillBatch::prepare(&mut model, vec![(first, SamplingLogits::Full)], None)?;
        assert!(!batch.execute_step(&mut model, 8)?.complete);
        assert!(model.sessions.is_empty());
        assert_eq!(model.prefill_page_headroom()?, 0);
        let second = request(&model, 1, 1);
        let output = model.prefill_with_budget(
            second.session_id,
            &second.prompt_tokens,
            &[],
            SamplingLogits::Full,
            None,
            second.generation_tokens,
            &mut |_| {},
        )?;
        let _token = admission::token(&model, output.output)?;
        assert_eq!(finish(&batch, &mut model, 32)?, [expected]);
        let sessions = model.sessions.keys().copied().collect::<Vec<_>>();
        for session in sessions {
            model.release_session(session)?;
        }
        model.clear_prefix_cache();
        assert_eq!(model.stream.paged_arenas().resident_arenas()?, 0);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
fn decode_reclaims_another_sessions_unused_tail_before_capacity_failure() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = bounded(family)?;
        let first = request(&model, 0, 1);
        let output = model.prefill_with_budget(
            first.session_id,
            &first.prompt_tokens,
            &[],
            SamplingLogits::Full,
            None,
            first.generation_tokens,
            &mut |_| {},
        )?;
        let token = admission::token(&model, output.output)?;
        let reference_id = Uuid::new_v4();
        let reference = model.sessions[&first.session_id].snapshot()?;
        model.sessions.insert(reference_id, reference);
        let expected = trajectory(&mut model, reference_id, token)?;
        model.release_session(reference_id)?;
        let second = request(&model, 1, usize::MAX);
        let output = model.prefill_with_budget(
            second.session_id,
            &second.prompt_tokens,
            &[],
            SamplingLogits::Full,
            None,
            second.generation_tokens,
            &mut |_| {},
        )?;
        let _token = admission::token(&model, output.output)?;
        assert_eq!(model.prefill_page_headroom()?, 0);
        assert_eq!(trajectory(&mut model, first.session_id, token)?, expected);
        assert!(model.prefill_page_headroom()? > 0, "decode did not reclaim the unused tail");
        model.release_session(first.session_id)?;
        model.release_session(second.session_id)?;
        model.clear_prefix_cache();
        assert_eq!(model.stream.paged_arenas().resident_arenas()?, 0);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

pub(super) fn trajectory(
    model: &mut LoadedModel,
    session: Uuid,
    mut token: u32,
) -> Result<Vec<u32>> {
    let mut tokens = Vec::new();
    for _ in 0..32 {
        let output = model.decode(session, token, SamplingLogits::Full)?;
        token = admission::token(model, output)?;
        tokens.push(token);
    }
    Ok(tokens)
}
