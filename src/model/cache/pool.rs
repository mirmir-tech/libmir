use std::{
    collections::HashMap,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use foundation::model::BackendTarget;
use runtime::kv::{CacheConfig, KvCache, KvCacheDType};

use crate::ModelMemoryEstimate;

#[derive(Debug)]
pub(in crate::model) struct SharedKvCache {
    pub(in crate::model) cache: Mutex<KvCache>,
    pub(in crate::model) ready: Condvar,
    config: CacheConfig,
    memory_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) struct SharedCacheMemory {
    pub(in crate::model) id: u64,
    pub(in crate::model) bytes: u64,
}

pub(in crate::model) struct CacheAssignment {
    pub(in crate::model) cache: Arc<SharedKvCache>,
    pub(in crate::model) config: CacheConfig,
    pub(in crate::model) shared_memory: Option<SharedCacheMemory>,
}

#[derive(Clone, Debug, Default)]
pub(in crate::model) struct KvCachePools {
    pools: Arc<Mutex<HashMap<CacheKey, Weak<SharedKvCache>>>>,
    next_memory_id: Arc<AtomicU64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    block_size: usize,
    dtype: KvCacheDType,
    bytes_per_token: u64,
}

impl KvCachePools {
    /// The cache a model loads with. On Metal, automatically sized models of
    /// one K/V geometry share logical blocks. A model given its own context
    /// (`fixed`) keeps exactly that capacity, so it never shares.
    pub(in crate::model) fn acquire(
        &self,
        target: &BackendTarget,
        config: CacheConfig,
        estimate: ModelMemoryEstimate,
        fixed: bool,
    ) -> CacheAssignment {
        let Some(key) = shared_key(target, config, estimate, fixed) else {
            return independent(config);
        };
        let mut pools = self.pools.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        pools.retain(|_, cache| cache.strong_count() > 0);
        let cache = pools.get(&key).and_then(Weak::upgrade).unwrap_or_else(|| {
            let memory_id = self.next_memory_id.fetch_add(1, Ordering::Relaxed);
            let cache = Arc::new(SharedKvCache {
                cache: Mutex::new(KvCache::with_config(config)),
                ready: Condvar::new(),
                config,
                memory_id,
            });
            pools.insert(key, Arc::downgrade(&cache));
            cache
        });
        drop(pools);
        let config = cache.config;
        let bytes = estimate
            .kv_bytes_per_token
            .saturating_mul(u64::from(config.block_count))
            .saturating_mul(u64::try_from(config.block_size).unwrap_or(u64::MAX));
        CacheAssignment {
            shared_memory: Some(SharedCacheMemory { id: cache.memory_id, bytes }),
            cache,
            config,
        }
    }

    /// The cache configuration `acquire` would give, without acquiring it.
    pub(in crate::model) fn preview(
        &self,
        target: &BackendTarget,
        config: CacheConfig,
        estimate: ModelMemoryEstimate,
        fixed: bool,
    ) -> CacheConfig {
        let Some(key) = shared_key(target, config, estimate, fixed) else {
            return config;
        };
        let pools = self.pools.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        pools.get(&key).and_then(Weak::upgrade).map_or(config, |cache| cache.config)
    }
}

fn shared_key(
    target: &BackendTarget,
    config: CacheConfig,
    estimate: ModelMemoryEstimate,
    fixed: bool,
) -> Option<CacheKey> {
    (*target == BackendTarget::Metal && estimate.kv_bytes_per_token > 0 && !fixed).then_some(
        CacheKey {
            block_size: config.block_size,
            dtype: config.dtype,
            bytes_per_token: estimate.kv_bytes_per_token,
        },
    )
}

fn independent(config: CacheConfig) -> CacheAssignment {
    CacheAssignment {
        cache: Arc::new(SharedKvCache {
            cache: Mutex::new(KvCache::with_config(config)),
            ready: Condvar::new(),
            config,
            memory_id: 0,
        }),
        config,
        shared_memory: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_metal_models_share_logical_blocks() {
        let pools = KvCachePools::default();
        let first =
            pools.acquire(&BackendTarget::Metal, CacheConfig::new(10), estimate(1_024), false);
        let preview =
            pools.preview(&BackendTarget::Metal, CacheConfig::new(5), estimate(1_024), false);
        let second =
            pools.acquire(&BackendTarget::Metal, CacheConfig::new(5), estimate(1_024), false);

        assert!(Arc::ptr_eq(&first.cache, &second.cache));
        assert_eq!(second.config.block_count, 10);
        assert_eq!(preview.block_count, 10);
        assert_eq!(first.shared_memory, second.shared_memory);
    }

    #[test]
    fn a_model_given_its_own_context_keeps_it() {
        let pools = KvCachePools::default();
        let first =
            pools.acquire(&BackendTarget::Metal, CacheConfig::new(64), estimate(1_024), true);
        let preview =
            pools.preview(&BackendTarget::Metal, CacheConfig::new(128), estimate(1_024), true);
        let second =
            pools.acquire(&BackendTarget::Metal, CacheConfig::new(128), estimate(1_024), true);
        let automatic =
            pools.acquire(&BackendTarget::Metal, CacheConfig::new(32), estimate(1_024), false);

        assert!(!Arc::ptr_eq(&first.cache, &second.cache));
        assert_eq!((first.config.block_count, second.config.block_count), (64, 128));
        assert_eq!(preview.block_count, 128);
        assert!(second.shared_memory.is_none());
        assert_eq!(automatic.config.block_count, 32);
    }

    #[test]
    fn incompatible_geometry_and_cuda_keep_separate_blocks() {
        let pools = KvCachePools::default();
        let first =
            pools.acquire(&BackendTarget::Metal, CacheConfig::new(10), estimate(1_024), false);
        let other =
            pools.acquire(&BackendTarget::Metal, CacheConfig::new(10), estimate(2_048), false);
        let cuda =
            pools.acquire(&BackendTarget::Cuda, CacheConfig::new(10), estimate(1_024), false);

        assert!(!Arc::ptr_eq(&first.cache, &other.cache));
        assert!(!Arc::ptr_eq(&first.cache, &cuda.cache));
        assert!(cuda.shared_memory.is_none());
    }

    fn estimate(bytes_per_token: u64) -> ModelMemoryEstimate {
        ModelMemoryEstimate {
            weight_bytes: 0,
            kv_cache_bytes: 0,
            workspace_bytes: 0,
            required_bytes: 0,
            kv_bytes_per_token: bytes_per_token,
            cache_capacity_tokens: 0,
            model_context_tokens: 0,
            session_state_bytes: 0,
        }
    }
}
