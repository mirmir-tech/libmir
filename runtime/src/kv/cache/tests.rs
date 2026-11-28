use foundation::model::ModelId;

use super::*;
use crate::kv::{KvCacheDType, KvQuantMode};

fn model(name: &str) -> ModelId {
    ModelId::new(name)
}

#[test]
fn committed_prefix_is_reused() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 4,
        block_count: 2,
        dtype: KvCacheDType::Auto,
    });
    let gemma = model("gemma");
    let block = cache.allocate()?;
    let hash = cache.commit_prefix_block(&gemma, None, block, &[1, 2, 3, 4])?;
    let probe = cache.probe_prefix(&gemma, &[1, 2, 3, 4, 5]);

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
    let gemma = model("gemma");
    let block = cache.allocate()?;
    let hash = cache.commit_prefix_block(&gemma, None, block, &[1, 2])?;

    assert!(!cache.release(block)?);
    assert_eq!(cache.block_ref_count(block)?, 1);
    assert_eq!(cache.probe_prefix(&gemma, &[1, 2]).cached_blocks, vec![block]);
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
    let gemma = model("gemma");
    let first = cache.allocate()?;
    cache.commit_prefix_block(&gemma, None, first, &[1, 2])?;
    let second = cache.allocate()?;
    cache.commit_prefix_block(&gemma, None, second, &[1, 2])?;

    assert_eq!(cache.block_ref_count(first)?, 1);
    assert_eq!(cache.probe_prefix(&gemma, &[1, 2]).cached_blocks, [second]);
    assert!(cache.release(first)?);
    assert_eq!(cache.probe_prefix(&gemma, &[1, 2]).cached_blocks, [second]);
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
    let gemma = model("gemma");
    let oldest = cache.allocate()?;
    let oldest_hash = cache.commit_prefix_block(&gemma, None, oldest, &[1, 2])?;
    cache.release(oldest)?;
    let newest = cache.allocate()?;
    cache.commit_prefix_block(&gemma, None, newest, &[3, 4])?;
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
    let gemma = model("gemma");
    let block = cache.allocate()?;
    cache.commit_prefix_block(&gemma, None, block, &[1, 2])?;
    cache.retain(block)?;

    assert!(matches!(cache.allocate(), Err(RuntimeError::KvCachePressure)));
    assert_eq!(cache.block_ref_count(block)?, 3);
    assert_eq!(cache.stats().counters.protected_prefix_skips, 1);
    Ok(())
}

#[test]
fn impossible_allocation_is_typed_and_preserves_cached_prefix() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 1,
        dtype: KvCacheDType::Auto,
    });
    let id = model("model");
    let block = cache.allocate()?;
    cache.commit_prefix_block(&id, None, block, &[1, 2])?;
    cache.release(block)?;
    assert!(matches!(
        cache.allocate_for_tokens(3),
        Err(RuntimeError::KvCapacity { requested: 2, capacity: 1 })
    ));
    assert_eq!(cache.stats().counters.evictions, 0);
    assert_eq!(cache.probe_prefix(&id, &[1, 2]).cached_blocks, [block]);
    assert_eq!(cache.allocate_for_tokens(2)?.table.blocks(), &[block]);
    Ok(())
}

/// Freed blocks are handed out lowest first, so a backend indexing storage
/// by `BlockId` needs it only up to the blocks in use.
#[test]
fn allocates_the_lowest_free_block_first() -> Result<()> {
    let mut cache = KvCache::new(8);
    let blocks = (0..4).map(|_| cache.allocate()).collect::<Result<Vec<_>>>()?;
    assert_eq!(blocks, [BlockId(0), BlockId(1), BlockId(2), BlockId(3)]);
    cache.free(BlockId(1))?;
    cache.free(BlockId(3))?;
    assert_eq!(cache.allocate()?, BlockId(1));
    assert_eq!(cache.allocate()?, BlockId(3));
    assert_eq!(cache.allocate()?, BlockId(4));
    Ok(())
}
