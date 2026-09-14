use super::{
    fixture::{Family, load_family},
    *,
};

fn request(model: &LoadedModel, length: usize) -> PrefillRequest {
    PrefillRequest {
        model: ModelHandle {
            id: model.info.manifest.id.clone(),
            backend: "metal".into(),
        },
        session_id: Uuid::new_v4(),
        prompt_tokens: (0..length).map(|i| u32::try_from(i % 64).unwrap_or_default()).collect(),
        cache_checkpoints: Vec::new(),
        block_table: BlockTable::with_block_size(16),
        cached_tokens: 0,
        generation_tokens: None,
        sampling_logits: SamplingLogits::None,
    }
}

#[test]
fn cancelling_prefill_rows_preserves_sibling_logits_and_reindexes_progress() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 0)?;
        let requests = [request(&model, 32), request(&model, 32), request(&model, 32)];
        let (batch, _) = MetalPrefillBatch::prepare(
            &mut model,
            requests.iter().cloned().map(|r| (r, SamplingLogits::Full)).collect(),
            None,
        )?;
        assert!(!batch.execute_step(&mut model, 12)?.complete);
        batch.cancel_sessions(&mut model, &[requests[0].session_id, requests[2].session_id])?;
        {
            let state = batch.inner.lock()?;
            let rows = &state
                .as_ref()
                .ok_or_else(|| Error::InvalidPrefillBatch("batch missing".into()))?
                .sequences;
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].request.session_id, requests[1].session_id);
            assert_eq!(rows[0].position, 4);
            drop(state);
        }
        let step = batch.execute_step(&mut model, 4)?;
        assert_eq!(step.events.len(), 1);
        assert_eq!(step.events[0].0, 0);
        let survivor = finish_logits(&batch, &mut model)?;
        let control_request = request(&model, 32);
        let (control, _) = MetalPrefillBatch::prepare(
            &mut model,
            vec![(control_request, SamplingLogits::Full)],
            None,
        )?;
        assert_eq!(survivor, finish_logits(&control, &mut model)?);
        assert!(model.sessions.is_empty());
        assert!(model.retained_execution_states().is_none());
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
fn cancellation_retires_completed_and_unfinished_rows_and_all_handles_see_empty_batch() -> Result<()>
{
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 0)?;
        let requests = [request(&model, 1), request(&model, 32)];
        let (batch, _) = MetalPrefillBatch::prepare(
            &mut model,
            requests.iter().cloned().map(|r| (r, SamplingLogits::None)).collect(),
            None,
        )?;
        let handle = batch.clone();
        assert!(!batch.execute_step(&mut model, 8)?.complete);
        assert!(model.sessions.contains_key(&requests[0].session_id));
        let sessions = requests.iter().map(|r| r.session_id).collect::<Vec<_>>();
        batch.cancel_sessions(&mut model, &sessions)?;
        assert!(model.sessions.is_empty());
        assert!(handle.execute_step(&mut model, 8)?.complete);
        assert!(handle.finish()?.is_empty());
        assert!(model.retained_execution_states().is_none());
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
fn cancelled_cohort_lease_does_not_consume_siblings() -> Result<()> {
    use crate::native::prefill::MetalPrefillCohort;
    let (mut model, directory) = load_family(Family::Hybrid, 0)?;
    let requests = [request(&model, 32), request(&model, 32)];
    let cohort = MetalPrefillCohort::prepare(&mut model, &requests)?;
    cohort.discard(&[requests[0].session_id])?;
    assert!(cohort.take(std::iter::once(requests[0].session_id)).is_err());
    let (batch, _) = MetalPrefillBatch::prepare(
        &mut model,
        vec![(requests[1].clone(), SamplingLogits::None)],
        Some(&cohort),
    )?;
    assert_eq!(finish(&batch, &mut model, 8)?.len(), 1);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

#[test]
fn failed_cancellation_drain_retains_removed_state_until_recovery() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 0)?;
        let requests = [request(&model, 32), request(&model, 32)];
        let (batch, _) = MetalPrefillBatch::prepare(
            &mut model,
            requests.iter().cloned().map(|r| (r, SamplingLogits::None)).collect(),
            None,
        )?;
        assert!(!batch.execute_step(&mut model, 8)?.complete);
        crate::native::model::recovery::fail_next_drain();
        assert!(batch.cancel_sessions(&mut model, &[requests[0].session_id]).is_err());
        assert_eq!(model.retained_execution_states().map(<[_]>::len), Some(1));
        assert!(model.require_execution_ready().is_err());
        model.recover_execution()?;
        assert!(model.retained_execution_states().is_none());
        assert_eq!(finish(&batch, &mut model, 4)?.len(), 1);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

fn finish_logits(batch: &MetalPrefillBatch, model: &mut LoadedModel) -> Result<Vec<f32>> {
    while !batch.execute_step(model, 4)?.complete {}
    let mut rows = batch.finish()?;
    assert_eq!(rows.len(), 1);
    let row = rows.pop().ok_or_else(|| Error::InvalidPrefillBatch("missing sibling".into()))?;
    let NativeOutput::Logits(logits) = row.native.output else {
        return Err(Error::InvalidPrefillBatch("expected full logits".into()));
    };
    let values = logits.to_vec_f32(model.stream())?;
    model.release_session(row.request.session_id)?;
    model.flush_decode_graphs()?;
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(values)
}

#[test]
fn interruption_yields_between_graphs_without_changing_row_indices_or_budget() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 0)?;
        let requests = [request(&model, 32), request(&model, 32)];
        let (batch, _) = MetalPrefillBatch::prepare(
            &mut model,
            requests.iter().cloned().map(|r| (r, SamplingLogits::None)).collect(),
            None,
        )?;
        let step = batch.execute_step_until(&mut model, 128, &mut || true)?;
        assert!(!step.complete);
        assert!(step.events.is_empty(), "already-cancelled batch executed a graph");
        let mut graphs = 0;
        let step = batch.execute_step_until(&mut model, 128, &mut || {
            graphs += 1;
            graphs > 1
        })?;
        assert!(!step.complete, "interrupted step consumed the complete budget");
        assert_eq!(step.events.iter().map(|(row, _)| *row).collect::<Vec<_>>(), [0, 1]);
        batch.cancel_sessions(&mut model, &[requests[0].session_id])?;
        assert_eq!(finish(&batch, &mut model, 128)?.len(), 1);
        model.flush_decode_graphs()?;
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}
