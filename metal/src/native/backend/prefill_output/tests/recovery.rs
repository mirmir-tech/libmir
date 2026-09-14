use super::*;

pub(super) fn exercise(manifest: &ModelManifest) -> Result<()> {
    let mut config = crate::MetalConfig::default();
    config.tuning.mode = TuningMode::Disabled;
    config.cache.prefix_cache_entries = 0;
    config.set_max_batch_requests(2);
    let backend = MetalBackend::new(config);
    let model = backend.load_model_with_progress(manifest, &mut |_| {})?;
    let invalid = SamplingLogits::TopK { k: 1, vocab_size: 0 };
    let mut scalar = request(&model, 0);
    scalar.sampling_logits = invalid;
    assert!(backend.prefill_request_with_progress(&scalar, &mut |_| {}).is_err());
    let scalar_id = scalar.session_id;
    backend.with_model(&model.id, move |loaded| {
        assert!(
            !loaded.sessions.contains_key(&scalar_id),
            "materialization failure left an active scalar session"
        );
        Ok(())
    })?;
    let mut requests = [request(&model, 0), request(&model, 3)];
    requests[1].sampling_logits = invalid;
    let batch = backend.prepare_prefill_batch(&requests, None, &mut |_, _| {})?;
    assert!(
        backend
            .execute_generation_step(None, Some(&batch), 256, &mut |_, _| {})?
            .prefill?
    );
    let sessions = requests.map(|request| request.session_id);
    backend.with_model(&model.id, move |loaded| {
        assert!(sessions.iter().all(|session| loaded.sessions.contains_key(session)));
        Ok(())
    })?;
    assert!(backend.finish_prefill_batch(batch).is_err());
    backend.with_model(&model.id, move |loaded| {
        assert!(
            sessions.iter().all(|session| !loaded.sessions.contains_key(session)),
            "materialization failure left rows active"
        );
        Ok(())
    })?;
    let fresh = request(&model, 0);
    assert!(backend.prefill_request_with_progress(&fresh, &mut |_| {})?.next_token.is_some());
    backend.release_session(&model, fresh.session_id)?;
    assert!(backend.unload_model(&model)?);
    Ok(())
}
