use models::layout::AttentionLayerType;
use runtime::kv::KvCacheDType;

use super::LoadedModel;
use crate::{
    engine::{Array, DecoderCache, KvPageFormat, MemoryStats, clear_memory_cache, memory_stats},
    native::{error::Result, prefix::PrefixCache, session::SessionState},
};

const AUTO_PREFIX_CACHE_NUMERATOR: usize = 2;
const AUTO_PREFIX_CACHE_DIVISOR: usize = 5;
const ALLOCATOR_CACHE_DIVISOR: usize = 8;
const ALLOCATOR_PRESSURE_PERCENT: usize = 85;

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
        self.stream.synchronize()?;
        self.stream.detach_paged_arena_graphs()?;
        let _reclaimed = Self::reclaim_prefill_allocator_cache()?;
        Ok(())
    }

    pub(crate) fn flush_decode_graphs(&self) -> Result<()> {
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

    pub(crate) fn reserve_prefill_pages(&mut self, required: usize) -> Result<()> {
        let maximum = DecoderCache::physical_page_capacity(&self.stream, self.info.cache_step);
        let Some(decoder) = self.info.decoder.as_ref() else {
            return Ok(());
        };
        let Some(layer) = (0..decoder.num_hidden_layers)
            .find(|layer| decoder.layer_type(*layer) == AttentionLayerType::Full)
        else {
            return Ok(());
        };
        let kv_heads = decoder.layer_key_value_heads(layer);
        let head_dim = decoder.layer_head_dim(layer);
        let format = KvPageFormat::resolve(self.stream.config().kv_cache.dtype)?;
        let available = |model: &Self| {
            model
                .stream
                .paged_arenas()
                .available_pages(maximum, layer, kv_heads, head_dim, format)
        };
        let mut evicted = false;
        while available(self)? < required {
            if !self.prefixes.evict_oldest() {
                break;
            }
            evicted = true;
        }
        if evicted {
            clear_memory_cache()?;
        }
        let available = available(self)?;
        if available < required {
            return Err(crate::engine::Error::InvalidModel(format!(
                "Metal prefill requires {required} free K/V pages but only {available} remain"
            ))
            .into());
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

pub(in crate::native) fn cache_prefix_snapshot(
    prefixes: &mut PrefixCache,
    model: &str,
    tokens: &[u32],
    state: &SessionState,
    logits: &Array,
    block_size: Option<usize>,
    bytes: usize,
) -> Result<bool> {
    if !prefixes.enabled() {
        return Ok(false);
    }
    let _reclaimed = LoadedModel::reclaim_prefill_allocator_cache()?;
    prefixes.insert(model, tokens, state, logits, block_size, bytes)?;
    Ok(true)
}

pub(in crate::native) fn cache_prefix_checkpoint(
    prefixes: &mut PrefixCache,
    model: &str,
    tokens: &[u32],
    state: &SessionState,
    block_size: usize,
    bytes: usize,
) -> Result<bool> {
    if !prefixes.enabled() {
        return Ok(false);
    }
    let _reclaimed = LoadedModel::reclaim_prefill_allocator_cache()?;
    prefixes.insert_checkpoint(model, tokens, state, block_size, bytes)?;
    Ok(true)
}

fn usable_memory(memory: MemoryStats) -> usize {
    match (memory.limit, memory.recommended) {
        (0, None | Some(0)) => 0,
        (0, Some(recommended)) => recommended,
        (limit, None | Some(0)) => limit,
        (limit, Some(recommended)) => limit.min(recommended),
    }
}
