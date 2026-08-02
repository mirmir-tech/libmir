use super::*;
use crate::kv::{KvCacheDType, KvQuantMode};

#[test]
fn committed_prefix_is_reused() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 4,
        block_count: 2,
        dtype: KvCacheDType::Auto,
    });
    let block = cache.allocate()?;
    let hash = cache.commit_prefix_block("gemma", None, block, &[1, 2, 3, 4])?;
    let probe = cache.probe_prefix("gemma", &[1, 2, 3, 4, 5]);

    assert_eq!(probe.cached_blocks, vec![block]);
    assert_eq!(probe.cached_tokens, 4);
    assert_eq!(probe.missing_tokens, 1);
    assert_eq!(probe.last_hash, Some(hash));
    assert_eq!(cache.block_ref_count(block)?, 2);
    Ok(())
}

#[test]
fn stats_include_cache_dtype() {
    let cache = KvCache::with_config(CacheConfig {
        block_size: 8,
        block_count: 1,
        dtype: KvCacheDType::NvFp4,
    });
    let stats = cache.stats();

    assert_eq!(stats.dtype, KvCacheDType::NvFp4);
    assert_eq!(stats.quant_mode, KvQuantMode::NvFp4);
}

#[test]
fn release_decrements_shared_block_before_freeing() -> Result<()> {
    let mut cache = KvCache::new(1);
    let block = cache.allocate()?;
    cache.retain(block)?;

    assert_eq!(cache.block_ref_count(block)?, 2);
    assert!(!cache.release(block)?);
    assert_eq!(cache.block_ref_count(block)?, 1);
    assert!(cache.release(block)?);
    assert_eq!(cache.free_blocks(), 1);
    Ok(())
}

#[test]
fn committed_prefix_survives_request_release_until_evicted() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 1,
        dtype: KvCacheDType::Auto,
    });
    let block = cache.allocate()?;
    let hash = cache.commit_prefix_block("gemma", None, block, &[1, 2])?;

    assert!(!cache.release(block)?);
    assert_eq!(cache.block_ref_count(block)?, 1);
    assert_eq!(cache.probe_prefix("gemma", &[1, 2]).cached_blocks, vec![block]);
    assert!(cache.evict_prefix(hash)?);
    assert_eq!(cache.free_blocks(), 1);
    Ok(())
}

#[test]
fn duplicate_prefix_releases_the_replaced_cache_reference() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 2,
        dtype: KvCacheDType::Auto,
    });
    let first = cache.allocate()?;
    cache.commit_prefix_block("gemma", None, first, &[1, 2])?;
    let second = cache.allocate()?;
    cache.commit_prefix_block("gemma", None, second, &[1, 2])?;

    assert_eq!(cache.block_ref_count(first)?, 1);
    assert_eq!(cache.probe_prefix("gemma", &[1, 2]).cached_blocks, [second]);
    assert!(cache.release(first)?);
    assert_eq!(cache.probe_prefix("gemma", &[1, 2]).cached_blocks, [second]);
    assert!(!cache.release(second)?);
    assert_eq!(cache.free_blocks(), 1);
    Ok(())
}

#[test]
fn allocation_pressure_evicts_oldest_inactive_prefix() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 2,
        dtype: KvCacheDType::Auto,
    });
    let oldest = cache.allocate()?;
    let oldest_hash = cache.commit_prefix_block("gemma", None, oldest, &[1, 2])?;
    cache.release(oldest)?;
    let newest = cache.allocate()?;
    cache.commit_prefix_block("gemma", None, newest, &[3, 4])?;
    cache.release(newest)?;

    let allocation = cache.allocate_for_tokens(2)?;

    assert_eq!(allocation.table.blocks(), &[oldest]);
    assert!(cache.prefix.peek(oldest_hash).is_none());
    assert_eq!(cache.free_blocks(), 0);
    assert_eq!(cache.stats().counters.evictions, 1);
    Ok(())
}

#[test]
fn allocation_pressure_preserves_prefixes_used_by_sessions() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 1,
        dtype: KvCacheDType::Auto,
    });
    let block = cache.allocate()?;
    cache.commit_prefix_block("gemma", None, block, &[1, 2])?;
    cache.retain(block)?;

    assert!(cache.allocate().is_err());
    assert_eq!(cache.block_ref_count(block)?, 3);
    assert_eq!(cache.stats().counters.protected_prefix_skips, 1);
    Ok(())
}
