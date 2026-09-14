use std::sync::atomic::Ordering;

use super::{
    fixture::{Family, load_family},
    *,
};

mod execution;
mod scalar;
mod support;

fn request(model: &LoadedModel, length: usize) -> PrefillRequest {
    PrefillRequest {
        model: ModelHandle {
            id: model.info.manifest.id.clone(),
            backend: "metal".into(),
        },
        session_id: Uuid::new_v4(),
        prompt_tokens: (0..length).map(|i| u32::try_from(i % 64).unwrap_or(0)).collect(),
        cache_checkpoints: Vec::new(),
        block_table: BlockTable::with_block_size(16),
        cached_tokens: 0,
        generation_tokens: None,
        sampling_logits: SamplingLogits::None,
    }
}

#[test]
fn failed_packed_prefill_invalidates_every_handle() -> Result<()> {
    batch_failure(false)
}

#[test]
fn failed_mixed_prefill_retires_already_completed_rows() -> Result<()> {
    batch_failure(true)
}

fn batch_failure(mixed: bool) -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 0)?;
        let survivors = support::survivors(&mut model)?;
        let requests = [
            request(
                &model,
                if mixed {
                    1
                } else {
                    32
                },
            ),
            request(&model, 32),
        ];
        let inputs = requests.iter().cloned().map(|r| (r, SamplingLogits::None)).collect();
        let (batch, _) = MetalPrefillBatch::prepare(&mut model, inputs, None)?;
        let handle = batch.clone();
        let calls = execution::install(&mut model, execution::Stage::Prefill, 1);
        support::arm_drain_failure();
        assert!(matches!(
            batch.execute_step(&mut model, 8),
            Err(Error::ExecutionRecoveryFailed { .. })
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(batch.inner.lock()?.is_none(), "failed batch remained reusable");
        assert!(handle.execute_step(&mut model, 8).is_err());
        assert!(handle.finish().is_err());
        for request in requests {
            assert!(!model.sessions.contains_key(&request.session_id));
        }
        drop(batch);
        drop(handle);
        support::recover(&mut model, 2, survivors)?;
        support::continue_and_release(&mut model, survivors)?;
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}
