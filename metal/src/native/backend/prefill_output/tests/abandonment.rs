use super::*;

pub(super) fn exercise(manifest: &ModelManifest) -> Result<()> {
    let mut config = crate::MetalConfig::default();
    config.tuning.mode = TuningMode::Disabled;
    config.cache.prefix_cache_entries = 0;
    config.cache.paged_attention_min_context = 1;
    config.set_max_batch_requests(2);
    let backend = MetalBackend::new(config);
    let model = backend.load_model_with_progress(manifest, &mut |_| {})?;
    let mut requests = [request(&model, 0), request(&model, 3)];
    requests[0].prompt_tokens.truncate(1);
    let batch = backend.prepare_prefill_batch(&requests, None, &mut |_, _| {})?;
    let other = batch.clone();
    assert!(
        !backend
            .execute_generation_step(None, Some(&batch), 8, &mut |_, _| {})?
            .prefill?
    );
    let first = requests[0].session_id;
    drop(batch);
    backend.with_model(&model.id, move |loaded| {
        assert!(loaded.sessions.contains_key(&first), "dropping one clone cancelled the group");
        Ok(())
    })?;
    // The last handle can disappear inside a model task: cleanup must enqueue,
    // not synchronously wait for the same worker.
    backend.with_model(&model.id, move |_| {
        crate::native::model::recovery::fail_next_drain();
        drop(other);
        Ok(())
    })?;
    assert_eq!(backend.model_client(&model.id)?.retained_execution_state_count()?, Some(2));
    backend.with_model(&model.id, move |loaded| {
        assert!(
            !loaded.sessions.contains_key(&first),
            "abandoned prefill left its completed row active"
        );
        assert_eq!(loaded.stream().paged_arenas().resident_arenas()?, 0);
        Ok(())
    })?;
    let fresh = request(&model, 0);
    assert!(backend.prefill_request_with_progress(&fresh, &mut |_| {})?.next_token.is_some());
    backend.release_session(&model, fresh.session_id)?;
    verify_finished_clone(&backend, &model)?;
    let requests = [request(&model, 0), request(&model, 3)];
    let batch = backend.prepare_prefill_batch(&requests, None, &mut |_, _| {})?;
    assert!(
        !backend
            .execute_generation_step(None, Some(&batch), 8, &mut |_, _| {})?
            .prefill?
    );
    assert!(backend.unload_model(&model)?);
    drop(batch);
    assert_eq!(backend.paged_arenas.resident_arenas()?, 0);
    Ok(())
}

fn verify_finished_clone(backend: &MetalBackend, model: &ModelHandle) -> Result<()> {
    let requests = [request(model, 0), request(model, 3)];
    let batch = backend.prepare_prefill_batch(&requests, None, &mut |_, _| {})?;
    let other = batch.clone();
    assert!(
        backend
            .execute_generation_step(None, Some(&batch), 256, &mut |_, _| {})?
            .prefill?
    );
    assert_eq!(backend.finish_prefill_batch(batch)?.len(), 2);
    drop(other);
    let sessions = requests.map(|request| request.session_id);
    backend.with_model(&model.id, move |loaded| {
        assert!(
            sessions.iter().all(|session| loaded.sessions.contains_key(session)),
            "dropping a finished handle cancelled live sessions"
        );
        Ok(())
    })?;
    for session in sessions {
        backend.release_session(model, session)?;
    }
    Ok(())
}
