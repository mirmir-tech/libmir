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
    pub(in crate::model) fn acquire(
        &self,
        target: &BackendTarget,
        config: CacheConfig,
        estimate: ModelMemoryEstimate,
    ) -> CacheAssignment {
        if *target != BackendTarget::Metal || estimate.kv_bytes_per_token == 0 {
            return independent(config);
        }
        let key = CacheKey {
            block_size: config.block_size,
            dtype: config.dtype,
            bytes_per_token: estimate.kv_bytes_per_token,
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
        let first = pools.acquire(&BackendTarget::Metal, CacheConfig::new(10), estimate(1_024));
        let second = pools.acquire(&BackendTarget::Metal, CacheConfig::new(5), estimate(1_024));

        assert!(Arc::ptr_eq(&first.cache, &second.cache));
        assert_eq!(second.config.block_count, 10);
        assert_eq!(first.shared_memory, second.shared_memory);
    }

    #[test]
    fn incompatible_geometry_and_cuda_keep_separate_blocks() {
        let pools = KvCachePools::default();
        let first = pools.acquire(&BackendTarget::Metal, CacheConfig::new(10), estimate(1_024));
        let other = pools.acquire(&BackendTarget::Metal, CacheConfig::new(10), estimate(2_048));
        let cuda = pools.acquire(&BackendTarget::Cuda, CacheConfig::new(10), estimate(1_024));

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
        }
    }
}
