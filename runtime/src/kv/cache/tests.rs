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
