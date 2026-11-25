use models::layout::{AttentionLayerType, DecoderConfig};

use super::LoadedModel;
use crate::{
    engine::{DecoderCache, KvPageFormat, clear_memory_cache},
    native::error::Result,
};
impl LoadedModel {
    pub(crate) fn reserve_prefill_pages(&mut self, required: usize) -> Result<()> {
        self.require_execution_ready()?;
        if required == 0 {
            return Ok(());
        }
        let maximum = DecoderCache::physical_page_capacity(&self.stream, self.info.cache_step);
        let Some(decoder) = self.info.decoder.as_ref() else {
            return Ok(());
        };
        let Some(first_layer) = paged_layers(decoder).next() else {
            return Ok(());
        };
        if required > maximum {
            return Err(crate::engine::Error::KvPageCapacity {
                layer: first_layer,
                required,
                maximum,
            }
            .into());
        }
        let mut evicted = false;
        let mut available = self.prefill_page_headroom()?;
        if available < required
            && self.stream.config().cache.decode_reservation
                == crate::config::DecodeReservation::GenerationBudget
        {
            self.reclaim_decode_reservations()?;
            available = self.prefill_page_headroom()?;
        }
        while available < required {
            if !self.prefixes.evict_oldest() {
                break;
            }
            evicted = true;
            available = self.prefill_page_headroom()?;
        }
        if evicted {
            clear_memory_cache()?;
        }
        if available < required {
            return Err(crate::engine::Error::InvalidModel(format!(
                "Metal prefill requires {required} free K/V pages per full-attention layer but the minimum available is {available}"
            ))
            .into());
        }
        Ok(())
    }

    pub(in crate::native) fn reclaim_decode_reservations(&mut self) -> Result<()> {
        self.prefill_reservations.reclaim()?;
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        for state in self.sessions.values_mut() {
            state
                .cache
                .release_reservation_after(state.position.saturating_add(page_size))?;
        }
        Ok(())
    }

    pub(in crate::native) fn prefill_page_headroom(&self) -> Result<usize> {
        let maximum = DecoderCache::physical_page_capacity(&self.stream, self.info.cache_step);
        self.min_full_layer_pages(|layer, kv_heads, head_dim, format| {
            self.stream
                .paged_arenas()
                .available_pages(maximum, layer, kv_heads, head_dim, format)
        })
    }

    /// Free pages every full-attention layer already holds memory for.
    pub(in crate::native) fn allocated_page_headroom(&self) -> Result<usize> {
        self.min_full_layer_pages(|layer, kv_heads, head_dim, format| {
            self.stream
                .paged_arenas()
                .allocated_free_pages(layer, kv_heads, head_dim, format)
        })
    }

    fn min_full_layer_pages(
        &self,
        pages: impl Fn(usize, usize, usize, KvPageFormat) -> crate::engine::Result<usize>,
    ) -> Result<usize> {
        let Some(decoder) = self.info.decoder.as_ref() else {
            return Ok(usize::MAX);
        };
        let format = KvPageFormat::resolve(self.stream.config().kv_cache.dtype)?;
        paged_layers(decoder).try_fold(usize::MAX, |available, layer| {
            let free = pages(
                layer,
                decoder.layer_key_value_heads(layer),
                decoder.layer_head_dim(layer),
                format,
            )?;
            Ok(available.min(free))
        })
    }
}

/// The full-attention layers with a paged arena of their own. A layer
/// attending over an earlier layer's keys and values (Gemma 4's E models)
/// holds no pages, so its empty arena must not bound the others' headroom.
pub(super) fn paged_layers(decoder: &DecoderConfig) -> impl Iterator<Item = usize> + '_ {
    (0..decoder.num_hidden_layers).filter(|&layer| {
        decoder.layer_type(layer) == AttentionLayerType::Full
            && decoder.key_value_source(layer).is_none()
    })
}
