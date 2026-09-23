//! Final K/V cache sizing of a loaded model.
//!
//! With automatic sizing on a backend that can reallocate pages, a model
//! loads and warms up with a provisional cache and is resized afterwards from
//! measured memory. Explicit `kv_blocks` and other backends keep the cache
//! resolved before loading.

use foundation::model::BackendTarget;
use runtime::kv::{CacheConfig, KvCache};

mod rebalance;
mod registry;

pub(in crate::model) use registry::ModelRegistry;

use super::{Model, ModelInner, automatic_cache};
use crate::{ModelMemoryEstimate, Result, RuntimeConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KvSizing {
    /// The cache resolved before loading is final.
    Fixed,
    /// A provisional cache carries warm-up; the final size is measured then
    /// and may reach `capacity` blocks, which bounds one sequence's table.
    Provisional { blocks: u32, capacity: u32 },
    /// The cache was resized from measured memory.
    Measured(automatic_cache::MeasuredKvBudget),
}

impl KvSizing {
    /// Replaces an automatic estimate with a provisional cache where the
    /// backend can resize after warm-up.
    pub(super) fn for_load(
        library: &RuntimeConfig,
        resolved: &mut RuntimeConfig,
        estimate: ModelMemoryEstimate,
        target: &BackendTarget,
    ) -> Self {
        if !library.automatic_kv_cache
            || *target != BackendTarget::Cuda
            || estimate.kv_bytes_per_token == 0
        {
            return Self::Fixed;
        }
        let capacity = resolved.kv_cache.block_count;
        let blocks = automatic_cache::provisional_blocks(resolved, capacity);
        resolved.kv_cache.block_count = blocks;
        Self::Provisional { blocks, capacity }
    }

    /// Blocks one sequence's table must be able to address.
    pub(super) const fn sequence_capacity_blocks(self) -> Option<u32> {
        match self {
            Self::Provisional { capacity, .. } => Some(capacity),
            Self::Fixed | Self::Measured(_) => None,
        }
    }
}

impl Model {
    /// The K/V cache configuration this model serves with.
    #[must_use]
    pub fn kv_cache_config(&self) -> CacheConfig {
        self.inner
            .cache
            .cache
            .lock()
            .map_or(self.inner.config.kv_cache, |cache| CacheConfig {
                block_count: u32::try_from(cache.stats().total_blocks).unwrap_or(u32::MAX),
                ..self.inner.config.kv_cache
            })
    }

    /// Sizes a provisional cache from the memory left after warm-up. When the
    /// result cannot serve one long request, idle models loaded earlier give
    /// up pages down to an equal share first.
    pub(super) fn finalize_kv_cache(&self) -> Result<()> {
        let KvSizing::Provisional { blocks: provisional, capacity } = self.kv_sizing()? else {
            return Ok(());
        };
        // Warm-up left freed shapes cached in the allocator; return them so
        // the measurement sees what is really in use.
        self.inner.engine.settle_resident_memory(&self.inner.handle)?;
        let Some(mut measured) = self.measure_kv_budget(provisional)? else {
            return self.set_kv_sizing(KvSizing::Fixed);
        };
        if rebalance::below_minimum_share(&self.inner.config, measured)
            && rebalance::shrink_others(self, measured)?
            && let Some(again) = self.measure_kv_budget(provisional)?
        {
            measured = again;
        }
        // Sequence tables were sized for `capacity` at load.
        measured.max_blocks = measured.max_blocks.min(capacity);
        if measured.blocks > capacity {
            let block_bytes = measured.page_bytes / u64::from(measured.blocks.max(1));
            measured.blocks = capacity;
            measured.page_bytes = block_bytes.saturating_mul(u64::from(capacity));
            measured.checkpoint_bytes = checkpoint_bytes(measured.page_bytes);
        }
        measured.others_bytes = self.inner.memory.others_bytes()?;
        if measured.blocks != provisional && !self.resize_to(measured.blocks)? {
            return self.set_kv_sizing(KvSizing::Fixed);
        }
        self.set_kv_sizing(KvSizing::Measured(measured))?;
        tracing::info!(
            model = %self.inner.handle.id,
            provisional_blocks = provisional,
            blocks = measured.blocks,
            capacity_tokens = u64::from(measured.blocks)
                .saturating_mul(u64::try_from(self.inner.config.kv_cache.block_size).unwrap_or(u64::MAX)),
            page_bytes = measured.page_bytes,
            checkpoint_bytes = measured.checkpoint_bytes,
            budget_bytes = measured.budget_bytes,
            available_bytes = measured.available_bytes,
            "measured model KV cache"
        );
        Ok(())
    }

    /// Measures the budget while `current_blocks` pages are allocated.
    pub(super) fn measure_kv_budget(
        &self,
        current_blocks: u32,
    ) -> Result<Option<automatic_cache::MeasuredKvBudget>> {
        let target = self.inner.engine.target();
        let estimate = self.inner.descriptor.memory_estimate_for(&self.inner.config, &target);
        let memory = self.inner.engine.memory_snapshot()?;
        let headroom = self.inner.engine.retention_headroom_bytes(&self.inner.handle)?;
        // A measured session costs what its recurrent state, checkpoint and
        // attention workspace took; the estimate is only a floor.
        let session_bytes = self
            .inner
            .engine
            .session_bytes(&self.inner.handle)?
            .map_or(estimate.session_state_bytes, |measured| {
                measured.max(estimate.session_state_bytes)
            });
        tracing::debug!(
            model = %self.inner.handle.id,
            retention_headroom_bytes = headroom,
            session_bytes,
            available_bytes = memory.available_bytes,
            "measuring K/V budget"
        );
        Ok(automatic_cache::measured(
            &self.inner.config,
            estimate,
            &memory,
            current_blocks,
            automatic_cache::TrafficSlack {
                retention_headroom_bytes: headroom,
                session_bytes,
            },
        ))
    }

    /// Reallocates the cache for `blocks`. The host block lock is held across
    /// the backend resize, so admission waits and no session can start on the
    /// old pages; the backend refuses while sessions exist. `Ok(false)` when
    /// the backend keeps its pages.
    pub(super) fn resize_to(&self, blocks: u32) -> Result<bool> {
        let cache = CacheConfig {
            block_count: blocks,
            ..self.inner.config.kv_cache
        };
        let resized = self.with_cache(|host| {
            if !self.inner.engine.resize_kv_cache(&self.inner.handle, cache)? {
                return Ok(false);
            }
            // Blocks still held here are retained prefixes whose device state
            // went with the old pages.
            let stats = host.stats();
            tracing::debug!(
                model = %self.inner.handle.id,
                dropped_prefixes = stats.cached_prefixes,
                dropped_blocks = stats.used_blocks,
                blocks,
                "replaced host K/V blocks for the resized cache"
            );
            *host = KvCache::with_config(cache);
            Ok(true)
        })?;
        if !resized {
            return Ok(false);
        }
        self.inner.engine.settle_resident_memory(&self.inner.handle)?;
        self.inner.coordinator.reprofile(
            &self.inner.engine,
            &self.inner.handle,
            &self.inner.config.scheduler,
            cache,
        )?;
        let target = self.inner.engine.target();
        self.inner.memory.resize(self.inner.descriptor.memory_estimate_for(
            &RuntimeConfig {
                kv_cache: cache,
                ..self.inner.config.clone()
            },
            &target,
        ))?;
        Ok(true)
    }

    pub(super) fn kv_sizing(&self) -> Result<KvSizing> {
        self.inner.kv_sizing.lock().map(|sizing| *sizing).map_err(|_| {
            crate::RuntimeError::Config("model K/V sizing lock is poisoned".into()).into()
        })
    }

    pub(super) fn set_kv_sizing(&self, sizing: KvSizing) -> Result<()> {
        self.inner.kv_sizing.lock().map_or_else(
            |_| Err(crate::RuntimeError::Config("model K/V sizing lock is poisoned".into()).into()),
            |mut current| {
                *current = sizing;
                Ok(())
            },
        )
    }
}

/// The backend's prefix checkpoint budget for `page_bytes` of pages.
pub(super) fn checkpoint_bytes(page_bytes: u64) -> u64 {
    usize::try_from(page_bytes).map_or(page_bytes / 4, |bytes| {
        u64::try_from(CacheConfig::prefix_checkpoint_bytes(bytes)).unwrap_or(u64::MAX)
    })
}
