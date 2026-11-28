use runtime::kv::KvStorageSpec;

use super::{CudaSharedRoutedLayerState, CudaSharedRoutedModelTemplate, SharedRoutedLayerTemplate};
use crate::{DecodeAttentionBf16, Error, PagedKvCache, Result};

impl CudaSharedRoutedModelTemplate {
    /// K/V geometry of layer `index`, or `None` for a recurrent layer.
    pub(super) fn layer_storage(
        &self,
        index: usize,
        layer: &SharedRoutedLayerTemplate,
    ) -> Option<KvStorageSpec> {
        match layer {
            SharedRoutedLayerTemplate::Linear(_) | SharedRoutedLayerTemplate::Mamba2(_) => None,
            SharedRoutedLayerTemplate::Full(_) => Some(KvStorageSpec::new(
                self.cache,
                self.decoder.layer_key_value_heads(index),
                self.decoder.layer_head_dim(index),
            )),
            SharedRoutedLayerTemplate::Attention(layer) => Some(layer.storage(self.cache)),
            SharedRoutedLayerTemplate::Latent(layer) => Some(layer.storage(self.cache)),
        }
    }

    pub(super) fn cache_spec(&self) -> Result<KvStorageSpec> {
        self.layers
            .iter()
            .enumerate()
            .find_map(|(index, layer)| self.layer_storage(index, layer))
            .ok_or(Error::InvalidPagedKv("shared-routed model has no attention cache"))
    }

    /// One cache per decoder layer, then the prediction module's.
    pub(crate) fn allocate_shared_kv(&self) -> Result<Vec<Option<PagedKvCache>>> {
        self.backend.allocate_cache_layout(
            &self.cache_layout,
            self.cache,
            1,
            self.prediction.is_some(),
        )
    }

    /// The prediction module's attention over its pages, when loaded.
    pub(super) fn prepare_prediction_state(
        &self,
        caches: &[Option<PagedKvCache>],
    ) -> Result<Option<DecodeAttentionBf16>> {
        self.prediction
            .as_ref()
            .map(|prediction| {
                let cache = caches
                    .get(self.layers.len())
                    .cloned()
                    .flatten()
                    .ok_or(Error::InvalidPagedKv("prediction module cache is missing"))?;
                prediction.layer().prepare_state_with_cache(cache)
            })
            .transpose()
    }

    /// Whether every layer keeps its sequence state in the shared K/V pages
    /// (attention and latent attention, no recurrence or prediction module),
    /// so a session continues a cached prefix from its pages alone.
    #[must_use]
    pub(crate) fn pages_hold_state(&self) -> bool {
        self.prediction.is_none()
            && self.layers.iter().all(|layer| {
                matches!(
                    layer,
                    SharedRoutedLayerTemplate::Full(_) | SharedRoutedLayerTemplate::Latent(_)
                )
            })
    }

    pub(super) fn prepare_states(
        &self,
        caches: &[Option<PagedKvCache>],
    ) -> Result<Vec<CudaSharedRoutedLayerState>> {
        let expected = self.layers.len() + usize::from(self.prediction.is_some());
        if caches.len() != expected {
            return Err(Error::InvalidPagedKv("shared-routed cache count differs from layers"));
        }
        let missing = || Error::InvalidPagedKv("shared-routed attention cache is missing");
        self.layers
            .iter()
            .zip(caches)
            .map(|(layer, cache)| {
                Ok(match layer {
                    SharedRoutedLayerTemplate::Linear(layer) => {
                        CudaSharedRoutedLayerState::Linear(layer.prepare_state()?)
                    },
                    SharedRoutedLayerTemplate::Full(layer) => {
                        CudaSharedRoutedLayerState::Full(Box::new(layer.prepare_state_with_cache(
                            cache.clone().ok_or_else(missing)?,
                            self.max_sequence_blocks,
                        )?))
                    },
                    SharedRoutedLayerTemplate::Mamba2(layer) => {
                        CudaSharedRoutedLayerState::Mamba2(layer.prepare_state()?)
                    },
                    SharedRoutedLayerTemplate::Attention(layer) => {
                        CudaSharedRoutedLayerState::Attention(Box::new(
                            layer.prepare_state_with_cache(cache.clone().ok_or_else(missing)?)?,
                        ))
                    },
                    SharedRoutedLayerTemplate::Latent(layer) => {
                        CudaSharedRoutedLayerState::Latent(Box::new(
                            layer.prepare_state_with_cache(cache.clone().ok_or_else(missing)?)?,
                        ))
                    },
                })
            })
            .collect()
    }
}
