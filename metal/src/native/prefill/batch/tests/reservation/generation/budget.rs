use super::*;

#[test]
fn budget256_preserves_clamped_partial_prefixes_under_greedy_pressure() -> Result<()> {
    exercise(Family::Clamped)
}

#[test]
fn budget256_preserves_hybrid_partial_prefixes_under_greedy_pressure() -> Result<()> {
    exercise(Family::Hybrid)
}

fn exercise(family: Family) -> Result<()> {
    let (mut model, directory) = bounded(family)?;
    let mut first = request(&model, 0, 256);
    first.prompt_tokens.push(32);
    let cold = prefill(&mut model, &first)?;
    assert!(
        model.prefill_page_headroom()? > 0,
        "cold first-token COW did not reclaim the tail"
    );
    let expected = pressure::trajectory(&mut model, first.session_id, cold)?;
    model.release_session(first.session_id)?;
    model.clear_prefix_cache();
    let packed = request_copy(&first);
    let packed_id = packed.session_id;
    let (cold_batch, _) =
        MetalPrefillBatch::prepare(&mut model, vec![(packed, SamplingLogits::None)], None)?;
    assert_eq!(finish(&cold_batch, &mut model, 64)?, [cold]);
    assert!(model.prefill_page_headroom()? > 0);
    assert_eq!(pressure::trajectory(&mut model, packed_id, cold)?, expected);
    model.release_session(packed_id)?;
    let prefixes = model.prefixes.group_count();
    assert!(prefixes > 0);
    let exact = request_copy(&first);
    let output = model.prefill_with_budget(
        exact.session_id,
        &exact.prompt_tokens,
        &[],
        SamplingLogits::None,
        None,
        exact.generation_tokens,
        &mut |_| {},
    )?;
    assert_eq!(output.prefix_cache_tokens, 33);
    let token = admission::token(&model, output.output)?;
    assert_eq!(token, cold);
    assert_eq!(
        model.prefill_page_headroom()?,
        0,
        "exact-hit admission should use available headroom"
    );
    let mut extension = request_copy(&first);
    extension.prompt_tokens.push(cold);
    let (batch, _) =
        MetalPrefillBatch::prepare(&mut model, vec![(extension, SamplingLogits::None)], None)?;
    {
        let guard = batch.inner.lock()?;
        let state = guard
            .as_ref()
            .ok_or_else(|| Error::InvalidPrefillBatch("missing batch".into()))?;
        let position = state.sequences[0].position;
        drop(guard);
        // Attention snapshots may restore the aligned32-token prefix; recurrent
        // snapshots retain the exact33-token state instead. Both must reuse it.
        let restored = match family {
            Family::Clamped => 32,
            Family::Hybrid => 33,
        };
        assert_eq!(position, restored);
    }
    assert_eq!(finish(&batch, &mut model, 64)?, [expected[0]]);
    assert!(model.prefill_page_headroom()? > 0);
    assert!(model.prefixes.group_count() >= prefixes);
    let actual = pressure::trajectory(&mut model, exact.session_id, token)?;
    assert_eq!(actual, expected, "pressure/COW changed the original trajectory");
    while let Some(session) = model.sessions.keys().next().copied() {
        model.release_session(session)?;
    }
    model.clear_prefix_cache();
    assert_eq!(model.stream.paged_arenas().resident_arenas()?, 0);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

fn request_copy(request: &PrefillRequest) -> PrefillRequest {
    let mut copy = request.clone();
    copy.session_id = Uuid::new_v4();
    copy
}

fn prefill(model: &mut LoadedModel, request: &PrefillRequest) -> Result<u32> {
    let output = model.prefill_with_budget(
        request.session_id,
        &request.prompt_tokens,
        &[],
        SamplingLogits::None,
        None,
        request.generation_tokens,
        &mut |_| {},
    )?;
    admission::token(model, output.output)
}
