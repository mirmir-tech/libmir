//! Device memory of the mixed-mixer execution: K/V page bytes, the slack
//! its retained shapes may still grow by, and page reallocation.

use super::{MixedMixerExecution, PrefixCheckpoints};
use crate::{Error, Result};

/// The prefix checkpoint store bounded by a share of the K/V page bytes
/// (`CacheConfig::prefix_checkpoint_bytes`).
pub(super) fn prefix_checkpoints(caches: &[Option<crate::PagedKvCache>]) -> PrefixCheckpoints {
    PrefixCheckpoints::new(128, runtime::kv::CacheConfig::prefix_checkpoint_bytes(kv_bytes(caches)))
}

/// Bytes of the K/V pages.
fn kv_bytes(caches: &[Option<crate::PagedKvCache>]) -> usize {
    caches
        .iter()
        .flatten()
        .fold(0_usize, |total, cache| total.saturating_add(cache.bytes()))
}

/// Pool bytes the retained plans and combined batches may still grow by.
pub(super) fn retention_headroom_bytes(execution: &MixedMixerExecution) -> Result<u64> {
    let (plans, per_token) = execution.template.plan_retention()?;
    Ok(plans.saturating_add(execution.combined.headroom_bytes(per_token)))
}

/// Reallocates the K/V pages for `cache.block_count`. Sessions, checkpoints
/// and captured decode graphs hold the old pages and go; decode and prefill
/// batches address pages through sessions and stay, so memory measured with
/// them resident stays representative.
pub(super) fn resize_kv_cache(
    execution: &mut MixedMixerExecution,
    cache: runtime::kv::CacheConfig,
) -> Result<bool> {
    let current = execution.template.cache_config();
    if cache.block_size != current.block_size || cache.dtype != current.dtype {
        return Err(Error::InvalidPagedKv("K/V resize changes block size or dtype"));
    }
    execution.sessions.clear();
    for batch in execution.decode_batches.values_mut() {
        batch.follow_cache_blocks(cache.block_count);
    }
    execution.template = execution.template.with_cache_blocks(cache.block_count);
    // The old pages go before the new ones are allocated: holding both at
    // once would spike memory by a whole cache.
    execution.caches = Vec::new();
    execution.caches = execution.template.allocate_shared_kv()?;
    execution.checkpoints = prefix_checkpoints(&execution.caches);
    execution.combined.follow_cache_blocks(cache.block_count);
    for batch in execution.prefill_batches.values_mut() {
        batch.follow_cache_blocks(cache.block_count);
    }
    Ok(true)
}
