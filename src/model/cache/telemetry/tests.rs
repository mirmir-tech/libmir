use foundation::model::BackendTarget;
use runtime::kv::CacheConfig;

use super::*;
use crate::{ModelMemoryEstimate, model::cache::pool::KvCachePools};

#[test]
fn observer_never_waits_for_cache_or_keeps_it_alive() {
    let assignment = KvCachePools::default().acquire(
        &BackendTarget::CpuReference,
        CacheConfig::new(4),
        ModelMemoryEstimate {
            weight_bytes: 0,
            kv_cache_bytes: 0,
            workspace_bytes: 0,
            required_bytes: 0,
            kv_bytes_per_token: 0,
            cache_capacity_tokens: 0,
            model_context_tokens: 0,
            session_state_bytes: 0,
        },
    );
    let reader = CacheStatsReader(Arc::downgrade(&assignment.cache));
    assert!(reader.snapshot().is_some());
    let guard = assignment.cache.cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(reader.snapshot().is_none(), "a busy cache must not block the sampler");
    drop(guard);
    assert!(reader.snapshot().is_some());
    drop(assignment);
    assert!(reader.snapshot().is_none(), "the observer must not retain the cache");
}
