use super::*;
use crate::kv::{CacheConfig, KvCacheDType};

#[test]
fn prefill_reuses_committed_prefix_blocks() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 4,
        dtype: KvCacheDType::Auto,
    });
    let mut first = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let first_prefill = first.prepare_prefill(&mut cache, &[1, 2])?;
    assert_eq!(first_prefill.missing_tokens, 2);
    assert_eq!(first.commit_ready_prefix_blocks(&mut cache)?, 1);

    let shared = first.table().blocks()[0];
    let mut second = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let second_prefill = second.prepare_prefill(&mut cache, &[1, 2, 3])?;

    assert_eq!(second_prefill.cached_tokens, 2);
    assert_eq!(second_prefill.missing_tokens, 1);
    assert_eq!(second_prefill.write_offset, 2);
    assert_eq!(second_prefill.table.blocks()[0], shared);
    assert_eq!(cache.block_ref_count(shared)?, 3);
    let counters = cache.stats().counters;
    assert_eq!(counters.probes, 2);
    assert_eq!(counters.hits, 1);
    assert_eq!(counters.misses, 2);
    assert_eq!(counters.hit_tokens, 2);
    assert_eq!(counters.miss_tokens, 3);
    assert_eq!(second.release(&mut cache)?, 1);
    assert_eq!(cache.block_ref_count(shared)?, 2);
    Ok(())
}

#[test]
fn prefill_pressure_preserves_selected_prefix_and_evicts_another() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 2,
        dtype: KvCacheDType::Auto,
    });
    let mut shared = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    shared.prepare_prefill_in_place(&mut cache, &[1, 2])?;
    shared.commit_ready_prefix_blocks(&mut cache)?;
    let shared_block = shared.table().blocks()[0];
    shared.release(&mut cache)?;
    let mut other = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    other.prepare_prefill_in_place(&mut cache, &[3, 4])?;
    other.commit_ready_prefix_blocks(&mut cache)?;
    other.release(&mut cache)?;

    let mut next = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let prefill = next.prepare_prefill(&mut cache, &[1, 2, 5])?;

    assert_eq!(prefill.cached_tokens, 2);
    assert_eq!(prefill.table.blocks()[0], shared_block);
    assert_eq!(cache.block_ref_count(shared_block)?, 2);
    Ok(())
}

#[test]
fn prefill_admission_does_not_pin_shared_prefix_blocks() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 4,
        dtype: KvCacheDType::Auto,
    });
    let mut cached = cached_session(&mut cache, &[1, 2, 3, 4, 5, 6])?;
    let shared = cached.table().blocks()[0];
    cached.release(&mut cache)?;
    assert_eq!(cache.block_ref_count(shared)?, 1);

    let mut next = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let admission = next.probe_prefill_admission(&cache, &[1, 2, 3, 4, 9, 10, 11, 12], 0)?;

    assert!(admission.needs_eviction);
    assert_eq!(admission.missing_tokens, 4);
    assert_eq!(cache.block_ref_count(shared)?, 1);
    let prefill =
        next.prepare_prefill_with_reserve_in_place(&mut cache, &[1, 2, 3, 4, 9, 10, 11, 12], 0)?;
    assert_eq!(prefill.cached_tokens, 4);
    Ok(())
}

#[test]
fn two_phase_prefill_pins_later_hit_before_miss_evicts() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 4,
        dtype: KvCacheDType::Auto,
    });
    let mut first = cached_session(&mut cache, &[1, 2, 3, 4])?;
    let first_blocks = first.table().blocks().to_vec();
    first.release(&mut cache)?;
    let mut second = cached_session(&mut cache, &[5, 6, 7, 8])?;
    let second_blocks = second.table().blocks().to_vec();
    second.release(&mut cache)?;

    let mut miss = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let miss_plan = miss.probe_prefill_with_reserve_in_place(&mut cache, &[9, 10, 11, 12], 0)?;
    assert!(miss_plan.needs_eviction);
    let mut hit = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let hit_plan = hit.probe_prefill_with_reserve_in_place(&mut cache, &[5, 6, 7, 8], 0)?;
    assert_eq!(hit_plan.cached_tokens, 4);

    let miss_step = miss.allocate_prefill_plan_in_place(&mut cache, miss_plan)?;
    let hit_step = hit.allocate_prefill_plan_in_place(&mut cache, hit_plan)?;

    assert_eq!(miss_step.cached_tokens, 0);
    assert_eq!(hit_step.cached_tokens, 4);
    assert_eq!(hit.table().blocks(), second_blocks);
    assert!(first_blocks.iter().all(|block| miss.table().blocks().contains(block)));
    Ok(())
}

#[test]
fn multimodal_prefill_never_reuses_or_publishes_token_only_prefixes() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 4,
        dtype: KvCacheDType::Auto,
    });
    let mut text = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    text.prepare_prefill_in_place(&mut cache, &[1, 2])?;
    assert_eq!(text.commit_ready_prefix_blocks(&mut cache)?, 1);

    let mut image = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let prefill = image.prepare_uncached_prefill_in_place(&mut cache, &[1, 2])?;
    assert_eq!(prefill.cached_tokens, 0);
    assert_eq!(prefill.missing_tokens, 2);
    assert_eq!(image.commit_ready_prefix_blocks(&mut cache)?, 0);
    image.append_decode_in_place(&mut cache, 3)?;
    assert_eq!(image.commit_ready_prefix_blocks(&mut cache)?, 0);
    Ok(())
}

#[test]
fn decode_append_allocates_only_on_block_boundary() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 4,
        dtype: KvCacheDType::Auto,
    });
    let mut state = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let _prefill = state.prepare_prefill(&mut cache, &[1, 2])?;

    let third = state.append_decode(&mut cache, 3)?;
    let fourth = state.append_decode(&mut cache, 4)?;

    assert_eq!(third.token_offset, 2);
    assert!(third.allocated_block.is_some());
    assert_eq!(fourth.token_offset, 3);
    assert_eq!(fourth.allocated_block, None);
    assert_eq!(state.token_len(), 4);
    Ok(())
}

#[test]
fn prefill_reserves_declared_decode_capacity() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 2,
        dtype: KvCacheDType::Auto,
    });
    let mut state = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    state.prepare_prefill_with_reserve_in_place(&mut cache, &[1, 2], 2)?;

    let third = state.append_decode_in_place(&mut cache, 3)?;
    let fourth = state.append_decode_in_place(&mut cache, 4)?;

    assert_eq!(state.table().blocks().len(), 2);
    assert_eq!(third.allocated_block, None);
    assert_eq!(fourth.allocated_block, None);
    assert_eq!(state.commit_ready_prefix_blocks(&mut cache)?, 2);
    Ok(())
}

#[test]
fn reserved_decode_slot_can_be_replaced_before_commit() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 4,
        dtype: KvCacheDType::Auto,
    });
    let mut state = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let _prefill = state.prepare_prefill(&mut cache, &[1])?;

    let reserved = state.reserve_decode_in_place(&mut cache)?;
    state.replace_token_at(reserved.token_offset, 2)?;

    assert_eq!(reserved.token_offset, 1);
    assert_eq!(state.token_len(), 2);
    assert_eq!(state.commit_ready_prefix_blocks(&mut cache)?, 1);
    Ok(())
}

#[test]
fn replacing_unknown_decode_slot_returns_error() {
    let mut state = KvSessionState::new(Uuid::new_v4(), "gemma", 2);

    assert!(state.replace_token_at(0, 9).is_err());
}

#[test]
fn decode_reservation_builds_single_token_write_plan() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 2,
        dtype: KvCacheDType::Auto,
    });
    let mut state = KvSessionState::new(Uuid::new_v4(), "gemma", 2);
    let _prefill = state.prepare_prefill(&mut cache, &[1])?;
    let decode = state.append_decode(&mut cache, 2)?;
    let plan = decode.write_plan(0)?;

    assert_eq!(plan.token_count(), 1);
    assert_eq!(plan.written_tokens(), 1);
    assert_eq!(plan.writes()[0].local_start, 0);
    assert_eq!(plan.writes()[0].local_end, 1);
    assert_eq!(plan.writes()[0].page_start, 1);
    assert_eq!(plan.writes()[0].page_end, 2);
    Ok(())
}

fn cached_session(cache: &mut KvCache, tokens: &[u32]) -> Result<KvSessionState> {
    let mut state = KvSessionState::new(Uuid::new_v4(), "gemma", cache.block_size());
    state.prepare_prefill_in_place(cache, tokens)?;
    state.commit_ready_prefix_blocks(cache)?;
    Ok(state)
}
