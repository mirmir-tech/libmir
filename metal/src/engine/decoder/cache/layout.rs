use std::sync::Arc;

use architecture::{CacheLayout, LayerCacheEntry, LayerCacheKind};

use super::{CacheStorage, DecoderCache, hybrid};
use crate::engine::{Error, KvCache, KvPageFormat, PagedArenaPool, Result};

impl DecoderCache {
    /// Validates that attention-only cache entries place every caching layer
    /// before any layer sharing an earlier layer's K/V cache.
    pub(crate) fn validate_attention_entries(entries: &[LayerCacheEntry]) -> Result<()> {
        let mut seen_shared = false;
        for entry in entries {
            match entry.kind {
                LayerCacheKind::SharedFrom { .. } => seen_shared = true,
                LayerCacheKind::Paged { .. } | LayerCacheKind::Ring { .. } => {
                    if seen_shared {
                        return Err(Error::CachingLayerAfterSharing { layer: entry.layer });
                    }
                },
                LayerCacheKind::Recurrent(_) => {
                    return Err(Error::InvalidModel(
                        "attention-only cache layout contains a recurrent layer".into(),
                    ));
                },
            }
        }
        Ok(())
    }

    pub(crate) fn from_layout(
        layout: &CacheLayout,
        step: usize,
        format: KvPageFormat,
        page_size: usize,
        max_pages: usize,
        pool: &Arc<PagedArenaPool>,
    ) -> Result<Self> {
        Self::from_entries(&layout.layers, step, format, page_size, max_pages, pool)
    }

    pub(crate) fn from_entries(
        entries: &[LayerCacheEntry],
        step: usize,
        format: KvPageFormat,
        page_size: usize,
        max_pages: usize,
        pool: &Arc<PagedArenaPool>,
    ) -> Result<Self> {
        let has_recurrent =
            entries.iter().any(|entry| matches!(entry.kind, LayerCacheKind::Recurrent(_)));
        if has_recurrent {
            let layers = hybrid::from_entries(entries, step, format, page_size, max_pages, pool)?;
            return Ok(Self {
                storage: CacheStorage::HybridLinear(layers),
            });
        }
        Self::validate_attention_entries(entries)?;
        let mut caches = Vec::with_capacity(entries.len());
        for entry in entries {
            match entry.kind {
                LayerCacheKind::Paged { .. } => {
                    caches.push(KvCache::new_paged_with_pool_capacity(
                        step,
                        page_size,
                        format,
                        max_pages,
                        Arc::clone(pool),
                        entry.layer,
                    )?);
                },
                LayerCacheKind::Ring { window, .. } => {
                    caches.push(KvCache::new_with_window(step, Some(window.get()))?);
                },
                LayerCacheKind::SharedFrom { .. } => {},
                LayerCacheKind::Recurrent(_) => {
                    unreachable!("recurrent entries handled above")
                },
            }
        }
        Ok(Self { storage: CacheStorage::Attention(caches) })
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use architecture::{CacheLayout, LayerCacheEntry, LayerCacheKind};
    use models::{
        layout::DecoderConfig,
        semantic::{LayerCacheShape, SemanticModelSpec},
        weights::{DenseDType, TensorCatalog, TensorInfo},
    };
    use serde_json::json;

    use super::*;

    #[test]
    fn chunked_llama4_layout_allocates_paged_caches_not_rings() -> Result<()> {
        let decoder = DecoderConfig::from_value(&json!({
            "model_type": "llama4_text", "hidden_size": 64, "intermediate_size": 32,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "vocab_size": 32, "num_local_experts": 4, "num_experts_per_tok": 1,
            "attention_chunk_size": 32, "attn_temperature_tuning": true, "floor_scale": 32,
            "attn_scale": 0.1, "use_qk_norm": true
        }))?;
        let catalog = TensorCatalog {
            tensors: vec![TensorInfo {
                name: "model.layers.0.input_layernorm.weight".into(),
                file: std::path::PathBuf::new(),
                container: models::layout::WeightContainer::SafeTensors,
                dtype: DenseDType::Bf16.into(),
                shape: vec![64],
                data_start: 0,
                data_offsets: [0, 0],
            }],
        };
        let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
        let layout = CacheLayout::for_metal(&spec.decoder);
        assert!(
            layout
                .layers
                .iter()
                .all(|entry| matches!(entry.kind, LayerCacheKind::Paged { .. }))
        );
        let pool = Arc::new(PagedArenaPool::default());
        let mut cache =
            DecoderCache::from_layout(&layout, 64, KvPageFormat::Native, 16, 64, &pool)?;
        assert_eq!(cache.attention_caches_mut()?.len(), 4);
        for layer in cache.attention_caches_mut()? {
            assert!(layer.window_newest_slot().is_none());
        }
        Ok(())
    }

    #[test]
    fn rejects_caching_layer_after_shared_layer() {
        let shape = LayerCacheShape { heads: 2, key_width: 16, value_width: 16 };
        let window = NonZeroUsize::new(8).unwrap_or(NonZeroUsize::MIN);
        let entries = [
            LayerCacheEntry {
                layer: 0,
                kind: LayerCacheKind::Ring { shape, window, slots_per_session: 1 },
            },
            LayerCacheEntry {
                layer: 1,
                kind: LayerCacheKind::SharedFrom { source_layer: 0 },
            },
            LayerCacheEntry {
                layer: 2,
                kind: LayerCacheKind::Paged { shape },
            },
        ];
        let error = DecoderCache::validate_attention_entries(&entries);
        assert!(matches!(error, Err(Error::CachingLayerAfterSharing { layer: 2 })));
    }
}
