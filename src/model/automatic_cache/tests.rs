use super::*;

const GIB: u64 = 1024 * 1024 * 1024;

#[test]
fn automatic_cache_uses_device_budget_and_model_shape() {
    let config = RuntimeConfig::default();
    let estimate = estimate(8 * GIB, GIB, 147_456, 262_144);
    let memory = memory(128 * GIB, 110 * GIB, 4 * GIB, true);

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
    let estimate = estimate(0, 0, 1024, 8192);
    let memory = memory(16 * GIB, 16 * GIB, 0, false);

    assert_eq!(resolve(&config, estimate, &memory, 0).kv_cache.block_count, 1234);
}

#[test]
fn automatic_cache_leaves_headroom_for_the_admission_snapshot() {
    let config = RuntimeConfig::default();
    let estimate = estimate(2 * GIB, GIB, 64 * 1024, 262_144);
    let memory = memory(16 * GIB, 15 * GIB, 0, false);

    let resolved = resolve(&config, estimate, &memory, 0);
    let cache_bytes = cache_bytes(&resolved, estimate);

    assert!(
        cache_bytes + estimate.required_bytes
            <= memory.available_bytes.unwrap_or_default() - ADMISSION_SNAPSHOT_HEADROOM_BYTES
    );
}

#[test]
fn metal_geometry_fits_the_full_depth_four_c10_cohort() {
    let mut config = RuntimeConfig::default();
    config.memory.reserve_percent = Some(1);
    let estimate = estimate(27_953_641_429, 2_795_364_142, 225_280, 131_072);
    let memory = memory(68_719_476_736, 55_662_788_608, 0, true);

    let resolved = resolve(&config, estimate, &memory, 0);
    let capacity = cache_tokens(&resolved);

    assert!(capacity >= 10 * (4_096 + 2_048 + 128));
    assert!(
        cache_bytes(&resolved, estimate) + estimate.required_bytes
            <= memory.available_bytes.unwrap_or_default()
                - memory_policy::platform_reserve(config.memory, &memory)
    );
}

#[test]
fn repaired_gx10_geometry_keeps_dynamic_prefill_headroom() {
    let config = RuntimeConfig::default();
    let estimate = estimate(41_829_561_328, 4_182_956_132, 49_152, 131_072);
    let available = 126_958_235_648;
    let memory = memory(130_596_184_064, available, 4 * GIB, true);

    let resolved = resolve(&config, estimate, &memory, 0);

    assert_eq!(resolved.kv_cache.block_count, 55_865);
    assert_eq!(cache_tokens(&resolved), 893_840);
    assert!(
        cache_bytes(&resolved, estimate)
            + estimate.required_bytes
            + memory_policy::transient_reserve(estimate, &memory)
            <= available - memory_policy::platform_reserve(config.memory, &memory)
    );
}

#[test]
fn windowed_gx10_geometry_spends_only_full_layer_bytes_per_token() {
    let config = RuntimeConfig::default();
    let growing_bytes_per_token = 24_576;
    let ring_bytes = 12 * 144 * 64 * 2_048;
    let configured_tokens = 4_096 * 16;
    let mut estimate = estimate(41_829_561_328, 4_182_956_132, growing_bytes_per_token, 131_072);
    estimate.kv_cache_bytes = growing_bytes_per_token * configured_tokens + ring_bytes;
    estimate.required_bytes = 47_679_753_300;
    estimate.cache_capacity_tokens = configured_tokens;
    let memory = memory(130_596_184_064, 126_958_235_648, 4 * GIB, true);

    let resolved = resolve(&config, estimate, &memory, 0);

    assert_eq!(resolved.kv_cache.block_count, 111_155);
    assert_eq!(cache_tokens(&resolved), 1_778_480);
}

const fn estimate(
    weight_bytes: u64,
    workspace_bytes: u64,
    kv_bytes_per_token: u64,
    model_context_tokens: u64,
) -> ModelMemoryEstimate {
    ModelMemoryEstimate {
        weight_bytes,
        kv_cache_bytes: 0,
        workspace_bytes,
        required_bytes: weight_bytes + workspace_bytes,
        kv_bytes_per_token,
        cache_capacity_tokens: 0,
        model_context_tokens,
    }
}

fn memory(total: u64, available: u64, reserve: u64, unified: bool) -> MemorySnapshot {
    MemorySnapshot {
        total_bytes: Some(total),
        available_bytes: Some(available),
        active_bytes: 0,
        cached_bytes: 0,
        allocation_reserve_bytes: reserve,
        source: "test".to_owned(),
        unified,
    }
}

fn cache_tokens(config: &RuntimeConfig) -> u64 {
    u64::from(config.kv_cache.block_count)
        * u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX)
}

fn cache_bytes(config: &RuntimeConfig, estimate: ModelMemoryEstimate) -> u64 {
    cache_tokens(config) * estimate.kv_bytes_per_token
}
