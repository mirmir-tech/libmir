use runtime::kv::{CacheStats, KvCache};

use super::Model;
use crate::Result;

impl Model {
    pub(crate) fn with_cache<T>(
        &self,
        use_cache: impl FnOnce(&mut KvCache) -> Result<T>,
    ) -> Result<T> {
        let Ok(mut cache) = self.inner.cache.lock() else {
            return Err(
                runtime::RuntimeError::KvCache("model KV cache lock is poisoned".into()).into()
            );
        };
        use_cache(&mut cache)
    }

    pub(crate) fn with_cache_wait<T>(
        &self,
        mut use_cache: impl FnMut(&mut KvCache) -> Result<T>,
    ) -> Result<T> {
        let Ok(mut cache) = self.inner.cache.lock() else {
            return Err(
                runtime::RuntimeError::KvCache("model KV cache lock is poisoned".into()).into()
            );
        };
        loop {
            match use_cache(&mut cache) {
                Err(crate::Error::Runtime(runtime::RuntimeError::KvCachePressure)) => {
                    let Ok(ready) = self.inner.cache_ready.wait(cache) else {
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
        self.inner.cache_ready.notify_all();
    }

    pub(crate) fn wait_for_cache_cohort(
        &self,
        needs_eviction: bool,
        missing_tokens: usize,
    ) -> std::time::Duration {
        self.inner.cache_cohort.wait(needs_eviction, missing_tokens)
    }

    #[must_use]
    /// Returns statistics for the K/V cache shared by this loaded model.
    pub fn cache_stats(&self) -> CacheStats {
        self.inner
            .cache
            .lock()
            .map_or_else(|poisoned| poisoned.into_inner().stats(), |cache| cache.stats())
    }
}
