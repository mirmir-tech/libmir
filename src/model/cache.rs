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

    #[must_use]
    /// Returns statistics for the K/V cache shared by this loaded model.
    pub fn cache_stats(&self) -> CacheStats {
        self.inner
            .cache
            .lock()
            .map_or_else(|poisoned| poisoned.into_inner().stats(), |cache| cache.stats())
    }
}
