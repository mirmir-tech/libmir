use models::layout::AttentionLayerType;
use runtime::kv::KvCacheDType;

use super::LoadedModel;
use crate::{
    engine::{Array, MemoryStats, clear_memory_cache, memory_stats},
    native::{error::Result, prefix::PrefixCache, session::SessionState},
};

const AUTO_PREFIX_CACHE_NUMERATOR: usize = 2;
const AUTO_PREFIX_CACHE_DIVISOR: usize = 5;
const ALLOCATOR_CACHE_DIVISOR: usize = 8;
const MEMORY_PRESSURE_PERCENT: usize = 85;

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
        let pressure = total > usable.saturating_mul(MEMORY_PRESSURE_PERCENT) / 100
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
    let mut memory = memory_stats()?;
    while !prefix_snapshot_fits(memory) && prefixes.evict_oldest() {
        clear_memory_cache()?;
        memory = memory_stats()?;
    }
    if !prefix_snapshot_fits(memory) {
        tracing::debug!(
            prefix_bytes = bytes,
            active_bytes = memory.active,
            cached_bytes = memory.cached,
            usable_bytes = usable_memory(memory),
            "skipped Metal prefix snapshot under memory pressure"
        );
        return Ok(false);
    }
    prefixes.insert(model, tokens, state, logits, block_size, bytes)?;
    Ok(true)
}

fn prefix_snapshot_fits(memory: MemoryStats) -> bool {
    let usable = usable_memory(memory);
    usable == 0
        || memory.active.saturating_add(memory.cached)
            <= usable.saturating_mul(MEMORY_PRESSURE_PERCENT) / 100
}

fn usable_memory(memory: MemoryStats) -> usize {
    match (memory.limit, memory.recommended) {
        (0, None | Some(0)) => 0,
        (0, Some(recommended)) => recommended,
        (limit, None | Some(0)) => limit,
        (limit, Some(recommended)) => limit.min(recommended),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_automatic_prefix_cache_to_two_fifths_of_usable_memory() {
        let memory = MemoryStats {
            active: 10,
            cached: 20,
            peak: 30,
            limit: 1_000,
            recommended: Some(800),
        };
        assert_eq!(prefix_cache_budget(memory, None), 320);
        assert_eq!(prefix_cache_budget(memory, Some(75)), 75);
    }

    #[test]
    fn prefix_snapshot_preserves_runtime_headroom() {
        let memory = MemoryStats {
            active: 700,
            cached: 50,
            peak: 750,
            limit: 1_000,
            recommended: Some(1_000),
        };
        assert!(prefix_snapshot_fits(memory));
        assert!(!prefix_snapshot_fits(MemoryStats { active: 801, ..memory }));
    }
}
