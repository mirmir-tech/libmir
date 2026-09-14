use super::*;
use crate::engine::DecoderCache;

#[test]
fn impossible_reservation_preserves_clamped_prefixes() -> Result<()> {
    exercise(Family::Clamped)
}

#[test]
fn impossible_reservation_preserves_hybrid_prefixes() -> Result<()> {
    exercise(Family::Hybrid)
}

fn exercise(family: Family) -> Result<()> {
    let (mut model, directory) = load_family(family, 10)?;
    let result = check(&mut model, family);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    result
}

fn check(model: &mut LoadedModel, family: Family) -> Result<()> {
    let prompts = [(0..32).collect::<Vec<u32>>(), (16..48).collect::<Vec<u32>>()];
    let mut expected = Vec::new();
    for prompt in &prompts {
        let session = Uuid::new_v4();
        let output =
            model.prefill(session, prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
        expected.push(token(model, output.output)?);
        model.release_session(session)?;
    }
    assert_eq!(model.prefixes.group_count(), 2);
    let maximum = DecoderCache::physical_page_capacity(model.stream(), model.info.cache_step);
    let required = maximum + 1;
    let rejected = model.reserve_prefill_pages(required);
    assert!(rejected.is_err());
    assert_eq!(
        model.prefixes.group_count(),
        2,
        "an impossible reservation evicted reusable prefixes"
    );
    let first_layer = match family {
        Family::Clamped => 0,
        Family::Hybrid => 1,
    };
    assert!(matches!(rejected, Err(Error::Engine(crate::engine::Error::KvPageCapacity {
        layer, required: requested, maximum: limit,
    })) if layer == first_layer && requested == required && limit == maximum));
    assert!(model.sessions.is_empty());
    assert!(model.retained_execution_states().is_none());
    for (prompt, expected) in prompts.iter().zip(expected) {
        let session = Uuid::new_v4();
        let output =
            model.prefill(session, prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
        assert_eq!(output.prefix_cache_tokens, prompt.len());
        assert_eq!(token(model, output.output)?, expected);
        model.release_session(session)?;
    }
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(())
}

#[test]
fn impossible_reserved_batch_preserves_clamped_prefixes() -> Result<()> {
    batch(Family::Clamped)
}

#[test]
fn impossible_reserved_batch_preserves_hybrid_prefixes() -> Result<()> {
    batch(Family::Hybrid)
}

fn batch(family: Family) -> Result<()> {
    let (model, directory) = load_family(family, 10)?;
    let manifest = model.info.manifest.clone();
    drop(model);
    let mut config = crate::MetalConfig::default();
    config.cache.prefix_cache_entries = 1;
    config.cache.paged_attention_min_context = 1;
    config.kv_cache.block_size = 16;
    config.kv_cache.block_count = 1;
    config.set_max_batch_requests(3);
    config.tuning.mode = runtime::tuning::TuningMode::Disabled;
    let mut model =
        LoadedModel::load_with_config(&manifest, std::sync::Arc::new(config), &mut |_| {})?;
    let result = check_batch(&mut model);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    result
}

fn check_batch(model: &mut LoadedModel) -> Result<()> {
    let prompt = (0..32).collect::<Vec<u32>>();
    let session = Uuid::new_v4();
    let output = model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
    let expected = token(model, output.output)?;
    model.release_session(session)?;
    let maximum = DecoderCache::physical_page_capacity(model.stream(), model.info.cache_step);
    assert!(maximum < 18, "fixture no longer constrains the three-row reservation");
    model.stream.set_decode_reservation(crate::config::DecodeReservation::Tokens(
        std::num::NonZeroUsize::new(64).ok_or(crate::engine::Error::ShapeOverflow)?,
    ));
    let requests = (1..=3)
        .map(|row| {
            (
                PrefillRequest {
                    model: ModelHandle {
                        id: model.info.manifest.id.clone(),
                        backend: "metal".into(),
                    },
                    session_id: Uuid::new_v4(),
                    prompt_tokens: (0..32).map(|i| (i + row) % 64).collect(),
                    cache_checkpoints: Vec::new(),
                    block_table: BlockTable::with_block_size(16),
                    cached_tokens: 0,
                    generation_tokens: None,
                    sampling_logits: SamplingLogits::None,
                },
                SamplingLogits::None,
            )
        })
        .collect();
    let rejected = MetalPrefillBatch::prepare(model, requests, None);
    assert!(
        matches!(rejected, Err(Error::Engine(crate::engine::Error::KvPageCapacity { required: 18, maximum: limit, .. })) if limit == maximum)
    );
    assert_eq!(model.prefixes.group_count(), 1);
    assert!(model.sessions.is_empty());
    assert!(model.retained_execution_states().is_none());
    model.stream.set_decode_reservation(crate::config::DecodeReservation::OnePage);
    let session = Uuid::new_v4();
    let output = model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
    assert_eq!(output.prefix_cache_tokens, prompt.len());
    assert_eq!(token(model, output.output)?, expected);
    model.release_session(session)?;
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(())
}

pub(super) fn token(model: &LoadedModel, output: NativeOutput) -> Result<u32> {
    Ok(match output {
        NativeOutput::Greedy(token) => token,
        NativeOutput::Logits(logits) => logits.argmax_u32(model.stream())?,
    })
}
