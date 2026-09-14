use models::layout::AttentionLayerType;
use runtime::kv::KvCacheDType;

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

#[cfg(test)]
mod history;
mod pages;
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
        let Some(decoder) = self.info.decoder.as_ref() else {
            return Ok(false);
        };
        let context = position.checked_add(sequence).ok_or(crate::engine::Error::ShapeOverflow)?;
        let mut largest_context = 0_usize;
        for layer in 0..decoder.num_hidden_layers {
            if decoder.layer_type(layer) != AttentionLayerType::Full {
                continue;
            }
            let elements = batch
                .checked_mul(context)
                .and_then(|value| value.checked_mul(decoder.layer_key_value_heads(layer)))
                .and_then(|value| value.checked_mul(decoder.layer_head_dim(layer)))
                .and_then(|value| value.checked_mul(2))
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
        self.stream.eval_many(&roots)?;
        self.stream.synchronize()?;
        self.stream.detach_paged_arena_graphs()?;
        for state in self.sessions.values() {
            state.cache.detach_evaluated_graphs(&self.stream)?;
        }
        Ok(())
    }

    pub(crate) fn estimated_prefix_bytes(&self, tokens: usize) -> Result<usize> {
        let Some(decoder) = self.info.decoder.as_ref() else {
            return Ok(0);
        };
        let dtype = self.stream.config().kv_cache.dtype;
        let bits = dtype.element_bits(16);
        let mut total = 0_usize;
        for layer in 0..decoder.num_hidden_layers {
            let length = match decoder.layer_type(layer) {
                AttentionLayerType::Linear => continue,
                AttentionLayerType::Full => tokens,
                AttentionLayerType::Sliding => {
                    tokens.min(decoder.layer_sliding_window(layer).unwrap_or(tokens))
                },
            };
            let elements = length
                .checked_mul(decoder.layer_key_value_heads(layer))
                .and_then(|value| value.checked_mul(decoder.layer_head_dim(layer)))
                .ok_or(crate::engine::Error::ShapeOverflow)?;
            let data_bits = elements
                .checked_mul(usize::from(bits.key) + usize::from(bits.value))
                .ok_or(crate::engine::Error::ShapeOverflow)?;
            let mut bytes = data_bits.div_ceil(8);
            if dtype == KvCacheDType::Int8PerTokenHead {
                let scales = length
                    .checked_mul(decoder.layer_key_value_heads(layer))
                    .and_then(|value| value.checked_mul(2 * size_of::<f32>()))
                    .ok_or(crate::engine::Error::ShapeOverflow)?;
                bytes = bytes.checked_add(scales).ok_or(crate::engine::Error::ShapeOverflow)?;
            }
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
