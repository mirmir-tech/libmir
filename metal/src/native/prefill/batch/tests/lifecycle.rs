use super::{
    fixture::{Family, load_family},
    *,
};

fn request(model: &LoadedModel) -> PrefillRequest {
    PrefillRequest {
        model: ModelHandle {
            id: model.info.manifest.id.clone(),
            backend: "metal".into(),
        },
        session_id: Uuid::new_v4(),
        prompt_tokens: (0..32).collect(),
        cache_checkpoints: Vec::new(),
        block_table: BlockTable::with_block_size(16),
        cached_tokens: 0,
        generation_tokens: None,
        sampling_logits: SamplingLogits::None,
    }
}

#[test]
fn premature_finish_preserves_progress_for_all_handles() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 0)?;
        let requests = (0..2)
            .map(|row| {
                let mut request = request(&model);
                if row == 0 {
                    request.prompt_tokens.truncate(1);
                }
                (request, SamplingLogits::None)
            })
            .collect();
        let (batch, _) = MetalPrefillBatch::prepare(&mut model, requests, None)?;
        let other_handle = batch.clone();
        assert!(!batch.execute_step(&mut model, 8)?.complete);
        assert_eq!(model.sessions.len(), 1, "first row must already be complete");
        assert!(batch.finish().is_err());
        assert_eq!(finish(&other_handle, &mut model, 8)?.len(), 2);
        assert!(batch.finish().is_err(), "completed batch must remain single-use");
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
fn duplicate_sessions_are_rejected_before_batch_preparation() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 0)?;
        let request = request(&model);
        let requests =
            vec![(request.clone(), SamplingLogits::None), (request, SamplingLogits::None)];
        assert!(MetalPrefillBatch::prepare(&mut model, requests, None).is_err());
        assert!(model.sessions.is_empty());
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
fn invalid_row_does_not_consume_other_rows_cohort_leases() -> Result<()> {
    use crate::native::prefill::MetalPrefillCohort;
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 0)?;
        let requests = [request(&model), request(&model)];
        let cohort = MetalPrefillCohort::prepare(&mut model, &requests)?;
        let mut invalid = requests[1].clone();
        invalid.prompt_tokens.clear();
        assert!(
            MetalPrefillBatch::prepare(
                &mut model,
                vec![(requests[0].clone(), SamplingLogits::None), (invalid, SamplingLogits::None)],
                Some(&cohort),
            )
            .is_err()
        );
        let valid = requests.into_iter().map(|r| (r, SamplingLogits::None)).collect();
        let (batch, _) = MetalPrefillBatch::prepare(&mut model, valid, Some(&cohort))?;
        assert_eq!(finish(&batch, &mut model, 8)?.len(), 2);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
fn absent_cohort_row_does_not_consume_valid_rows() -> Result<()> {
    use crate::native::prefill::MetalPrefillCohort;
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 0)?;
        let requests = [request(&model), request(&model)];
        let cohort = MetalPrefillCohort::prepare(&mut model, &requests)?;
        let absent = request(&model);
        assert!(
            MetalPrefillBatch::prepare(
                &mut model,
                vec![(requests[0].clone(), SamplingLogits::None), (absent, SamplingLogits::None)],
                Some(&cohort)
            )
            .is_err()
        );
        let valid = requests.into_iter().map(|r| (r, SamplingLogits::None)).collect();
        let (batch, _) = MetalPrefillBatch::prepare(&mut model, valid, Some(&cohort))?;
        assert_eq!(finish(&batch, &mut model, 8)?.len(), 2);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}
