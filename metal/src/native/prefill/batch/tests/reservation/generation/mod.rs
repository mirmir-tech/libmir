mod budget;
mod pressure;
use super::*;
use crate::{config::DecodeReservation, engine::DecoderCache};

fn bounded(family: Family) -> Result<(LoadedModel, std::path::PathBuf)> {
    let (model, directory) = load_family(family, 10)?;
    let manifest = model.info.manifest.clone();
    drop(model);
    let mut config = crate::MetalConfig::default();
    config.cache.decode_reservation = DecodeReservation::GenerationBudget;
    config.cache.prefix_cache_entries = 10;
    config.cache.paged_attention_min_context = 1;
    config.kv_cache.block_size = 16;
    config.kv_cache.block_count = 1;
    config.set_max_batch_requests(3);
    config.tuning.mode = runtime::tuning::TuningMode::Disabled;
    Ok((
        LoadedModel::load_with_config(&manifest, std::sync::Arc::new(config), &mut |_| {})?,
        directory,
    ))
}

fn request(model: &LoadedModel, seed: u32, budget: usize) -> PrefillRequest {
    PrefillRequest {
        model: ModelHandle {
            id: model.info.manifest.id.clone(),
            backend: "metal".into(),
        },
        session_id: Uuid::new_v4(),
        prompt_tokens: (0..32).map(|i| (i + seed) % 64).collect(),
        cache_checkpoints: Vec::new(),
        block_table: BlockTable::with_block_size(16),
        cached_tokens: 0,
        generation_tokens: std::num::NonZeroUsize::new(budget),
        sampling_logits: SamplingLogits::Full,
    }
}

#[test]
fn generation_budget_clamps_packed_admission_for_both_architectures() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = bounded(family)?;
        let capacity = DecoderCache::physical_page_capacity(&model.stream, model.info.cache_step);
        assert_eq!(capacity, 16);
        let requests = (0..3).map(|i| (request(&model, i, 64), SamplingLogits::Full)).collect();
        let (batch, _) = MetalPrefillBatch::prepare(&mut model, requests, None)?;
        {
            let guard = batch.inner.lock()?;
            let state = guard
                .as_ref()
                .ok_or_else(|| Error::InvalidPrefillBatch("missing batch".into()))?;
            assert_eq!(
                state.sequences.iter().map(|s| s.reservation.tokens()).collect::<Vec<_>>(),
                [80, 80, 96]
            );
            drop(guard);
        }
        let _tokens = finish(&batch, &mut model, 96)?;
        assert_eq!(model.sessions.len(), 3);
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
fn optional_tail_is_reclaimed_before_refill_evicts_prefixes() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = bounded(family)?;
        let first = request(&model, 0, usize::MAX);
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
        assert_eq!(model.prefill_page_headroom()?, 0);
        let second = request(&model, 1, 1);
        let (batch, _) =
            MetalPrefillBatch::prepare(&mut model, vec![(second, SamplingLogits::Full)], None)?;
        assert_eq!(model.prefixes.group_count(), 1, "optional tail evicted a reusable prefix");
        let _tokens = finish(&batch, &mut model, 64)?;
        assert_eq!(model.sessions.len(), 2);
        let restored = model
            .prefixes
            .lease_longest(&model.info.manifest.id, &first.prompt_tokens)?
            .ok_or(Error::NoPrefixLogits)?;
        let reference_id = Uuid::new_v4();
        model.sessions.insert(reference_id, restored.restored.0);
        let expected = model.decode(reference_id, token, SamplingLogits::Full)?;
        let expected = values(&model, expected)?;
        model.release_session(reference_id)?;
        let actual = model.decode(first.session_id, token, SamplingLogits::Full)?;
        assert_eq!(values(&model, actual)?, expected);
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
fn generation_budget_preserves_multimodel_prefix_churn() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 10)?;
        model.stream.set_decode_reservation(DecodeReservation::GenerationBudget);
        churn::exercise(&mut model)?;
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

fn finish(batch: &MetalPrefillBatch, model: &mut LoadedModel, budget: usize) -> Result<Vec<u32>> {
    for _ in 0..64 {
        if batch.execute_step(model, budget)?.complete {
            return batch
                .finish()?
                .into_iter()
                .map(|row| admission::token(model, row.native.output))
                .collect();
        }
    }
    Err(Error::InvalidPrefillBatch(
        "generation reservation prefill made no progress".into(),
    ))
}

fn values(model: &LoadedModel, output: NativeOutput) -> Result<Vec<f32>> {
    match output {
        NativeOutput::Logits(array) => Ok(array.to_vec_f32(model.stream())?),
        NativeOutput::Greedy(_) => Err(crate::engine::Error::ShapeOverflow.into()),
    }
}
