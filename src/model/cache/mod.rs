mod checkpoint;
mod pool;

pub(super) use pool::{KvCachePools, SharedCacheMemory, SharedKvCache};
use runtime::kv::{CacheStats, KvCache};

use super::{Model, cache_cohort::FillClaim};
use crate::Result;

impl Model {
    pub(crate) fn with_cache<T>(
        &self,
        use_cache: impl FnOnce(&mut KvCache) -> Result<T>,
    ) -> Result<T> {
        let Ok(mut cache) = self.inner.cache.cache.lock() else {
            return Err(
                runtime::RuntimeError::KvCache("model KV cache lock is poisoned".into()).into()
            );
        };
        use_cache(&mut cache)
    }

    pub(crate) fn with_cache_wait<T>(
        &self,
        use_cache: impl FnMut(&mut KvCache) -> Result<T>,
    ) -> Result<T> {
        self.with_cache_wait_cancellable(&crate::CancellationToken::default(), use_cache)
    }

    pub(crate) fn with_cache_wait_cancellable<T>(
        &self,
        cancellation: &crate::CancellationToken,
        mut use_cache: impl FnMut(&mut KvCache) -> Result<T>,
    ) -> Result<T> {
        let Ok(mut cache) = self.inner.cache.cache.lock() else {
            return Err(
                runtime::RuntimeError::KvCache("model KV cache lock is poisoned".into()).into()
            );
        };
        loop {
            cancellation.check()?;
            match use_cache(&mut cache) {
                Err(crate::Error::Runtime(runtime::RuntimeError::KvCachePressure)) => {
                    let Ok((ready, _)) = self
                        .inner
                        .cache
                        .ready
                        .wait_timeout(cache, crate::cancellation::POLL_INTERVAL)
                    else {
                        return Err(runtime::RuntimeError::KvCache(
                            "model KV cache wait is poisoned".into(),
                        )
                        .into());
                    };
                    cache = ready;
                },
                result => return result,
            }
        }
    }

    pub(crate) fn notify_cache_waiters(&self) {
        self.inner.cache.ready.notify_all();
    }

    pub(crate) fn wait_for_cache_cohort(
        &self,
        needs_eviction: bool,
        missing_tokens: usize,
        cancellation: &crate::CancellationToken,
    ) -> Result<std::time::Duration> {
        self.inner.cache_cohort.wait(needs_eviction, missing_tokens, cancellation)
    }

    pub(crate) fn claim_cache_fill(
        &self,
        tokens: &[u32],
        checkpoints: &[usize],
        cached_tokens: usize,
        cancellation: &crate::CancellationToken,
    ) -> Result<FillClaim> {
        self.inner
            .cache_cohort
            .claim_fill(tokens, checkpoints, cached_tokens, cancellation)
    }

    #[must_use]
    /// Returns statistics for the K/V cache shared by this loaded model.
    pub fn cache_stats(&self) -> CacheStats {
        self.inner
            .cache
            .cache
            .lock()
            .map_or_else(|poisoned| poisoned.into_inner().stats(), |cache| cache.stats())
    }
}
