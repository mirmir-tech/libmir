use runtime::kv::KvStorageSpec;

use super::{CudaSharedRoutedLayerState, CudaSharedRoutedModelTemplate, SharedRoutedLayerTemplate};
use crate::{Error, PagedKvCache, Result};

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
        }
    }

    pub(super) fn cache_spec(&self) -> Result<KvStorageSpec> {
        self.layers
            .iter()
            .enumerate()
            .find_map(|(index, layer)| self.layer_storage(index, layer))
            .ok_or(Error::InvalidPagedKv("shared-routed model has no attention cache"))
    }

    pub(crate) fn allocate_shared_kv(&self) -> Result<Vec<Option<PagedKvCache>>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                self.layer_storage(index, layer)
                    .map(|storage| self.backend.prepare_paged_kv(index, storage))
                    .transpose()
            })
            .collect()
    }

    pub(super) fn prepare_states(
        &self,
        caches: &[Option<PagedKvCache>],
    ) -> Result<Vec<CudaSharedRoutedLayerState>> {
        if caches.len() != self.layers.len() {
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
                })
            })
            .collect()
    }
}
