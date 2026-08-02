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
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn automatic_cache_uses_device_budget_and_model_shape() {
        let config = RuntimeConfig::default();
        let estimate = ModelMemoryEstimate {
            weight_bytes: 8 * GIB,
            kv_cache_bytes: 0,
            workspace_bytes: GIB,
            required_bytes: 9 * GIB,
            kv_bytes_per_token: 147_456,
            cache_capacity_tokens: 0,
            model_context_tokens: 262_144,
        };
        let memory = MemorySnapshot {
            total_bytes: Some(128 * GIB),
            available_bytes: Some(110 * GIB),
            active_bytes: 0,
            cached_bytes: 0,
            allocation_reserve_bytes: 4 * GIB,
            source: "test".to_owned(),
            unified: true,
        };

        let resolved = resolve(&config, estimate, &memory, 0);
        let shared = resolve(&config, estimate, &memory, 64 * GIB);

        assert!(resolved.kv_cache.block_count > 20_560);
        assert!(resolved.kv_cache.block_count < 50_000);
        assert!(shared.kv_cache.block_count < resolved.kv_cache.block_count);
    }

    #[test]
    fn explicit_block_count_is_preserved() {
        let config = RuntimeConfig {
            kv_cache: runtime::kv::CacheConfig::new(1234),
            automatic_kv_cache: false,
            ..RuntimeConfig::default()
        };
        let estimate = ModelMemoryEstimate {
            weight_bytes: 0,
            kv_cache_bytes: 0,
            workspace_bytes: 0,
            required_bytes: 0,
            kv_bytes_per_token: 1024,
            cache_capacity_tokens: 0,
            model_context_tokens: 8192,
        };
        let memory = MemorySnapshot {
            total_bytes: Some(16 * GIB),
            available_bytes: Some(16 * GIB),
            active_bytes: 0,
            cached_bytes: 0,
            allocation_reserve_bytes: 0,
            source: "test".to_owned(),
            unified: false,
        };

        assert_eq!(resolve(&config, estimate, &memory, 0).kv_cache.block_count, 1234);
    }

    #[test]
    fn automatic_cache_leaves_headroom_for_the_admission_snapshot() {
        let config = RuntimeConfig::default();
        let estimate = ModelMemoryEstimate {
            weight_bytes: 2 * GIB,
            kv_cache_bytes: 0,
            workspace_bytes: GIB,
            required_bytes: 3 * GIB,
            kv_bytes_per_token: 64 * 1024,
            cache_capacity_tokens: 0,
            model_context_tokens: 262_144,
        };
        let memory = MemorySnapshot {
            total_bytes: Some(16 * GIB),
            available_bytes: Some(15 * GIB),
            active_bytes: 0,
            cached_bytes: 0,
            allocation_reserve_bytes: 0,
            source: "test".to_owned(),
            unified: false,
        };

        let resolved = resolve(&config, estimate, &memory, 0);
        let cache_bytes = u64::from(resolved.kv_cache.block_count)
            * u64::try_from(resolved.kv_cache.block_size).unwrap_or(u64::MAX)
            * estimate.kv_bytes_per_token;

        assert!(
            cache_bytes + estimate.required_bytes
                <= memory.available_bytes.unwrap_or_default() - ADMISSION_SNAPSHOT_HEADROOM_BYTES
        );
    }

    #[test]
    fn repaired_gx10_geometry_keeps_dynamic_prefill_headroom() {
        let config = RuntimeConfig::default();
        let estimate = ModelMemoryEstimate {
            weight_bytes: 41_829_561_328,
            kv_cache_bytes: 0,
            workspace_bytes: 4_182_956_132,
            required_bytes: 46_012_517_460,
            kv_bytes_per_token: 49_152,
            cache_capacity_tokens: 0,
            model_context_tokens: 131_072,
        };
        let available = 126_958_235_648;
        let memory = MemorySnapshot {
            total_bytes: Some(130_596_184_064),
            available_bytes: Some(available),
            active_bytes: 0,
            cached_bytes: 0,
            allocation_reserve_bytes: 4 * GIB,
            source: "test".to_owned(),
            unified: true,
        };

        let resolved = resolve(&config, estimate, &memory, 0);
        let capacity = u64::from(resolved.kv_cache.block_count)
            * u64::try_from(resolved.kv_cache.block_size).unwrap_or(u64::MAX);

        assert_eq!(resolved.kv_cache.block_count, 29_271);
        assert_eq!(capacity, 468_336);
        assert!(
            u64::from(resolved.kv_cache.block_count)
                * u64::try_from(resolved.kv_cache.block_size).unwrap_or(u64::MAX)
                * estimate.kv_bytes_per_token
                + estimate.required_bytes
                + memory_policy::transient_reserve(estimate, &memory)
                <= available
                    .saturating_sub(memory_policy::platform_reserve(config.memory, &memory))
        );
    }

    #[test]
    fn windowed_gx10_geometry_spends_only_full_layer_bytes_per_token() {
        let config = RuntimeConfig::default();
        let growing_bytes_per_token = 24_576;
        let ring_bytes = 12 * 144 * 64 * 2_048;
        let configured_tokens = 4_096 * 16;
        let estimate = ModelMemoryEstimate {
            weight_bytes: 41_829_561_328,
            kv_cache_bytes: growing_bytes_per_token * configured_tokens + ring_bytes,
            workspace_bytes: 4_182_956_132,
            required_bytes: 47_679_753_300,
            kv_bytes_per_token: growing_bytes_per_token,
            cache_capacity_tokens: configured_tokens,
            model_context_tokens: 131_072,
        };
        let memory = MemorySnapshot {
            total_bytes: Some(130_596_184_064),
            available_bytes: Some(126_958_235_648),
            active_bytes: 0,
            cached_bytes: 0,
            allocation_reserve_bytes: 4 * GIB,
            source: "test".to_owned(),
            unified: true,
        };

        let resolved = resolve(&config, estimate, &memory, 0);

        assert_eq!(resolved.kv_cache.block_count, 57_966);
        assert_eq!(
            u64::from(resolved.kv_cache.block_count)
                * u64::try_from(resolved.kv_cache.block_size).unwrap_or(u64::MAX),
            927_456
        );
    }
}
