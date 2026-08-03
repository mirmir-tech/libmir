use runtime::kv::{CacheConfig, KvCacheDType};

use super::*;

#[test]
fn metal_reuses_an_engine_when_only_logical_capacity_changes() {
    let current = CacheConfig::new(1_000);
    let mut resolved = current;
    resolved.block_count = 500;

    assert!(engine_cache_compatible(&BackendTarget::Metal, resolved, current));
    assert!(!engine_cache_compatible(&BackendTarget::Cuda, resolved, current));
}

#[test]
fn metal_rebuilds_for_physical_cache_format_changes() {
    let current = CacheConfig::new(1_000);
    let mut resolved = current;
    resolved.dtype = KvCacheDType::Int8PerTokenHead;

    assert!(!engine_cache_compatible(&BackendTarget::Metal, resolved, current));
}

#[test]
fn metal_loads_the_resolved_logical_cache_capacity() {
    let mut config = RuntimeConfig::default();
    config.kv_cache.block_size = 16;
    config.kv_cache.block_count = 6_177;

    assert_eq!(resolved_metal_cache(&BackendTarget::Metal, &config), Some(config.kv_cache));
    assert_eq!(resolved_metal_cache(&BackendTarget::Cuda, &config), None);
}
