use super::memory_policy;
use crate::{MemorySnapshot, ModelMemoryEstimate, RuntimeConfig};

const UNIFIED_KV_BUDGET_NUMERATOR: u64 = 2;
const UNIFIED_KV_BUDGET_DIVISOR: u64 = 5;
const ADMISSION_SNAPSHOT_HEADROOM_BYTES: u64 = 64 * 1024 * 1024;

pub(super) fn resolve(
    config: &RuntimeConfig,
    estimate: ModelMemoryEstimate,
    memory: &MemorySnapshot,
    committed_bytes: u64,
) -> RuntimeConfig {
    if !config.automatic_kv_cache || estimate.kv_bytes_per_token == 0 {
        return config.clone();
    }
    let Some(mut available) =
        memory.available_bytes.map(|bytes| bytes.saturating_add(memory.cached_bytes))
    else {
        return config.clone();
    };
    if let Some(total) = memory.total_bytes {
        available = available.min(total.saturating_sub(committed_bytes));
    }
    let reserve = memory_policy::platform_reserve(config.memory, memory);
    let configured_tokens = u64::from(config.kv_cache.block_count)
        .saturating_mul(u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX));
    let fixed_kv = estimate
        .kv_cache_bytes
        .saturating_sub(estimate.kv_bytes_per_token.saturating_mul(configured_tokens));
    let fixed = estimate
        .weight_bytes
        .saturating_add(estimate.workspace_bytes)
        .saturating_add(fixed_kv);
    let remaining = available.saturating_sub(reserve).saturating_sub(fixed);
    let mut kv_budget = if memory.unified {
        remaining.saturating_sub(memory_policy::transient_reserve(estimate, memory))
    } else {
        remaining
    };
    // Memory availability is sampled again immediately before admission. Do not
    // let an automatically sized cache consume the entire first snapshot: even
    // allocator bookkeeping between the two samples would otherwise reject it.
    kv_budget = kv_budget.saturating_sub(ADMISSION_SNAPSHOT_HEADROOM_BYTES);
    if memory.unified
        && let Some(total) = memory.total_bytes
    {
        let unified_limit =
            (total / UNIFIED_KV_BUDGET_DIVISOR).saturating_mul(UNIFIED_KV_BUDGET_NUMERATOR);
        kv_budget = kv_budget.min(unified_limit);
    }
    let block_bytes = estimate
        .kv_bytes_per_token
        .saturating_mul(u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX));
    if block_bytes == 0 {
        return config.clone();
    }
    let memory_blocks = kv_budget / block_bytes;
    let useful_tokens = estimate
        .model_context_tokens
        .saturating_mul(u64::try_from(config.scheduler.max_batch_requests).unwrap_or(u64::MAX));
    let block_size = u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX);
    let useful_blocks = useful_tokens.div_ceil(block_size);
    let blocks = memory_blocks.min(useful_blocks).max(1).min(u64::from(u32::MAX));
    let mut resolved = config.clone();
    resolved.kv_cache.block_count = u32::try_from(blocks).unwrap_or(u32::MAX);
    resolved
}

#[cfg(test)]
mod tests;
