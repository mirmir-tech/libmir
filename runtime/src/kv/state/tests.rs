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
