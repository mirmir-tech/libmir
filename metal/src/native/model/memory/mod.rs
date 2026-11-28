use super::LoadedModel;
use crate::{
    engine::{MemoryStats, clear_memory_cache, memory_stats},
    native::{
        error::Result,
        prefill::diagnostics::{Stage, measure},
    },
};

const AUTO_PREFIX_CACHE_NUMERATOR: usize = 2;
const AUTO_PREFIX_CACHE_DIVISOR: usize = 5;
const ALLOCATOR_CACHE_DIVISOR: usize = 8;
const ALLOCATOR_PRESSURE_PERCENT: usize = 85;
const PACKED_PREFILL_RESERVE_DIVISOR: usize = 8;
const PACKED_PREFILL_MINIMUM_RESERVE: usize = 2 * 1024 * 1024 * 1024;
const PACKED_PREFILL_WORKSPACE_COPIES: usize = 8;

mod history;
mod prefix;
mod pressure;
pub(in crate::native) use prefix::{cache_prefix_checkpoint, cache_prefix_snapshot};
#[cfg(test)]
mod tests;

pub(super) fn prefix_cache_budget(memory: MemoryStats, configured: Option<usize>) -> usize {
    configured.unwrap_or_else(|| {
        let usable = usable_memory(memory);
        if usable == 0 {
            usize::MAX
        } else {
            usable / AUTO_PREFIX_CACHE_DIVISOR * AUTO_PREFIX_CACHE_NUMERATOR
        }
    })
}

impl LoadedModel {
    pub(crate) fn settle_prefill_graph(&self) -> Result<()> {
        measure(Stage::Synchronize, || Ok(self.stream.synchronize()?))?;
        measure(Stage::DetachArenas, || Ok(self.stream.detach_paged_arena_graphs()?))?;
        let _reclaimed = measure(Stage::Reclaim, Self::reclaim_prefill_allocator_cache)?;
        Ok(())
    }

    pub(crate) fn packed_prefill_fits(
        &self,
        batch: usize,
        position: usize,
        sequence: usize,
    ) -> Result<bool> {
        let memory = memory_stats()?;
        let usable = usable_memory(memory);
        if usable == 0 {
            return Ok(false);
        }
        let Some(layout) = self.info.cache_layout.as_ref() else {
            return Ok(false);
        };
        let context = position.checked_add(sequence).ok_or(crate::engine::Error::ShapeOverflow)?;
        let mut largest_context = 0_usize;
        // Every layer attending over the whole context, a latent one by the
        // row it caches; one sharing another's keys still attends.
        for (_, shape) in layout.paged_attention_layers() {
            let width = shape
                .key_width
                .checked_add(shape.value_width)
                .ok_or(crate::engine::Error::ShapeOverflow)?;
            let elements = batch
                .checked_mul(context)
                .and_then(|value| value.checked_mul(shape.heads))
                .and_then(|value| value.checked_mul(width))
                .ok_or(crate::engine::Error::ShapeOverflow)?;
            let bytes = elements
                .checked_mul(size_of::<f32>())
                .ok_or(crate::engine::Error::ShapeOverflow)?;
            largest_context = largest_context.max(bytes);
        }
        let workspace = largest_context
            .checked_mul(PACKED_PREFILL_WORKSPACE_COPIES)
            .ok_or(crate::engine::Error::ShapeOverflow)?;
        let reserve = (usable / PACKED_PREFILL_RESERVE_DIVISOR).max(PACKED_PREFILL_MINIMUM_RESERVE);
        let fits = memory
            .active
            .checked_add(workspace)
            .and_then(|used| used.checked_add(reserve))
            .is_some_and(|required| required <= usable);
        tracing::debug!(
            batch,
            position,
            sequence,
            active_bytes = memory.active,
            workspace_bytes = workspace,
            reserve_bytes = reserve,
            usable_bytes = usable,
            fits,
            "planned packed Metal prefill workspace"
        );
        Ok(fits)
    }

    pub(crate) fn flush_decode_graphs(&self) -> Result<()> {
        self.require_execution_ready()?;
        let mut roots = Vec::new();
        for state in self.sessions.values() {
            state.cache.extend_graph_roots(&mut roots);
        }
        // Persistent history leaves K/V page writes outside the output graph.
        self.stream.eval_many_with_paged_arenas(&roots)?;
        self.stream.synchronize()?;
        self.stream.detach_paged_arena_graphs()?;
        for state in self.sessions.values() {
            state.cache.detach_evaluated_graphs(&self.stream)?;
        }
        Ok(())
    }

    pub(crate) fn estimated_prefix_bytes(&self, tokens: usize) -> Result<usize> {
        let Some(contract) = self.info.contract.as_ref() else {
            return Ok(0);
        };
        let layout = architecture::CacheLayout::for_metal(&contract.semantic.decoder);
        let cache = self.stream.config().kv_cache;
        let mut total = 0_usize;
        for entry in &layout.layers {
            let (shape, length) = match entry.kind {
                architecture::LayerCacheKind::Paged { shape } => (shape, tokens),
                architecture::LayerCacheKind::Ring { shape, window, .. } => {
                    (shape, tokens.min(window.get()))
                },
                architecture::LayerCacheKind::SharedFrom { .. }
                | architecture::LayerCacheKind::Recurrent(_) => continue,
            };
            let budget = runtime::kv::KvStorageSpec {
                key_head_dim: shape.key_width,
                value_head_dim: shape.value_width,
                ..runtime::kv::KvStorageSpec::new(cache, shape.heads, shape.key_width)
            }
            .memory_budget();
            let bytes = length
                .checked_mul(budget.data_bytes_per_token + budget.scale_bytes_per_token)
                .ok_or(crate::engine::Error::ShapeOverflow)?;
            total = total.checked_add(bytes).ok_or(crate::engine::Error::ShapeOverflow)?;
        }
        Ok(total)
    }

    pub(crate) fn reclaim_prefill_allocator_cache() -> Result<bool> {
        let before = memory_stats()?;
        let usable = usable_memory(before);
        if usable == 0 {
            return Ok(false);
        }
        let total = before.active.saturating_add(before.cached);
        let pressure = total > usable.saturating_mul(ALLOCATOR_PRESSURE_PERCENT) / 100
            || before.cached > usable / ALLOCATOR_CACHE_DIVISOR;
        if !pressure {
            return Ok(false);
        }
        clear_memory_cache()?;
        let after = memory_stats()?;
        tracing::debug!(
            active_bytes = after.active,
            cached_bytes_before = before.cached,
            cached_bytes_after = after.cached,
            usable_bytes = usable,
            "reclaimed Metal allocator cache after prefill"
        );
        Ok(true)
    }
}

fn usable_memory(memory: MemoryStats) -> usize {
    match (memory.limit, memory.recommended) {
        (0, None | Some(0)) => 0,
        (0, Some(recommended)) => recommended,
        (limit, None | Some(0)) => limit,
        (limit, Some(recommended)) => limit.min(recommended),
    }
}
