use std::sync::{Arc, Weak};

use runtime::kv::CacheStats;

use super::SharedKvCache;
use crate::Model;

/// Read-only logical cache telemetry. Holds no model, engine, or device
/// resources.
#[derive(Clone, Debug)]
pub struct CacheStatsReader(Weak<SharedKvCache>);

impl CacheStatsReader {
    /// Returns no sample when the cache is busy or has been released.
    #[must_use]
    pub fn snapshot(&self) -> Option<CacheStats> {
        let cache = self.0.upgrade()?;
        let guard = cache.cache.try_lock().ok()?;
        Some(guard.stats())
    }
}

impl Model {
    /// Creates an observer that cannot keep this model or its accelerator
    /// alive.
    #[must_use]
    pub fn cache_stats_reader(&self) -> CacheStatsReader {
        CacheStatsReader(Arc::downgrade(&self.inner.cache))
    }
}

#[cfg(test)]
mod tests;
