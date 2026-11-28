use architecture::{CacheLayout, LayerCacheEntry, LayerCacheKind};
use models::semantic::LayerCacheShape;
use runtime::kv::{CacheConfig, KvStorageSpec};

use super::PagedKvCache;
use crate::{CudaBackend, Error, Result};

impl CudaBackend {
    /// Allocates every layer's K/V storage from the admitted `CacheLayout`.
    ///
    /// Returns one entry per decoder layer (plus prediction layers when
    /// `include_prediction` is set): `Some` for paged, ring, and shared-source
    /// layers, and `None` for recurrent layers.
    pub(crate) fn allocate_cache_layout(
        &self,
        layout: &CacheLayout,
        cache: CacheConfig,
        ring_sessions: usize,
        include_prediction: bool,
    ) -> Result<Vec<Option<PagedKvCache>>> {
        let total = layout.layers.len()
            + if include_prediction {
                layout.prediction_layers.len()
            } else {
                0
            };
        let ring_window = layout.max_ring_window();
        let mut caches: Vec<Option<PagedKvCache>> = Vec::with_capacity(total);
        let entries = layout.layers.iter().chain(
            include_prediction
                .then_some(layout.prediction_layers.as_slice())
                .into_iter()
                .flatten(),
        );
        for &entry in entries {
            let allocated =
                allocate_entry(self, entry, &caches, cache, ring_window, ring_sessions)?;
            caches.push(allocated);
        }
        Ok(caches)
    }
}

fn allocate_entry(
    backend: &CudaBackend,
    entry: LayerCacheEntry,
    existing: &[Option<PagedKvCache>],
    cache: CacheConfig,
    ring_window: Option<usize>,
    ring_sessions: usize,
) -> Result<Option<PagedKvCache>> {
    match entry.kind {
        LayerCacheKind::Paged { shape } => {
            Ok(Some(backend.prepare_paged_kv(entry.layer, storage_spec(cache, shape))?))
        },
        LayerCacheKind::Ring { shape, window, slots_per_session } => {
            let slots = ring_sessions
                .checked_mul(slots_per_session)
                .ok_or(Error::InvalidPagedKv("windowed KV slot capacity overflow"))?;
            let span = ring_window.unwrap_or_else(|| window.get());
            Ok(Some(backend.prepare_windowed_paged_kv(
                entry.layer,
                storage_spec(cache, shape),
                span,
                slots,
            )?))
        },
        LayerCacheKind::SharedFrom { source_layer } => {
            let shared = existing
                .get(source_layer)
                .and_then(Option::as_ref)
                .cloned()
                .ok_or(Error::InvalidDecoderKernel("shared keys come from a later layer"))?;
            Ok(Some(shared))
        },
        LayerCacheKind::Recurrent(_) => Ok(None),
    }
}

fn storage_spec(cache: CacheConfig, shape: LayerCacheShape) -> KvStorageSpec {
    KvStorageSpec {
        key_head_dim: shape.key_width,
        value_head_dim: shape.value_width,
        ..KvStorageSpec::new(cache, shape.heads, shape.key_width)
    }
}
