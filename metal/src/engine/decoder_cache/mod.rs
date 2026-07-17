use models::layout::AttentionLayerType;

use self::hybrid_cache::HybridLinearLayerCache;
use super::{Error, GatedDeltaState, KvCache, Result};

mod hybrid_cache;

#[derive(Debug)]
pub struct DecoderCache {
    storage: CacheStorage,
}

#[derive(Debug)]
enum CacheStorage {
    Attention(Vec<KvCache>),
    HybridLinear(Vec<HybridLinearLayerCache>),
}

impl DecoderCache {
    pub fn new(cache_windows: &[Option<usize>], step: usize) -> Result<Self> {
        let caches = cache_windows
            .iter()
            .map(|window| {
                if window.is_none() {
                    KvCache::new_paged(step, 16)
                } else {
                    KvCache::new_with_window(step, *window)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { storage: CacheStorage::Attention(caches) })
    }

    pub fn new_hybrid_linear(layer_types: &[AttentionLayerType], step: usize) -> Result<Self> {
        let layers = hybrid_cache::new(layer_types, step)?;
        Ok(Self {
            storage: CacheStorage::HybridLinear(layers),
        })
    }

    pub(crate) fn attention_caches_mut(&mut self) -> Result<&mut [KvCache]> {
        match &mut self.storage {
            CacheStorage::Attention(caches) => Ok(caches),
            CacheStorage::HybridLinear(_) => {
                Err(Error::InvalidModel("expected attention-only cache".into()))
            },
        }
    }

    pub(crate) fn gated_delta_state(&mut self, index: usize) -> Result<&mut GatedDeltaState> {
        match &mut self.storage {
            CacheStorage::HybridLinear(layers) => hybrid_cache::gated_delta_state(layers, index),
            CacheStorage::Attention(_) => {
                Err(Error::InvalidModel("expected hybrid linear cache".into()))
            },
        }
    }

    pub(crate) fn full_attention_cache(&mut self, index: usize) -> Result<&mut KvCache> {
        match &mut self.storage {
            CacheStorage::HybridLinear(layers) => hybrid_cache::full_attention_cache(layers, index),
            CacheStorage::Attention(_) => {
                Err(Error::InvalidModel("expected hybrid linear cache".into()))
            },
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        match &mut self.storage {
            CacheStorage::Attention(caches) => caches.iter_mut().try_for_each(KvCache::reset),
            CacheStorage::HybridLinear(layers) => hybrid_cache::reset(layers),
        }
    }

    pub fn reserve(&mut self, tokens: usize) -> Result<()> {
        match &mut self.storage {
            CacheStorage::Attention(caches) => {
                caches.iter_mut().try_for_each(|cache| cache.reserve(tokens))
            },
            CacheStorage::HybridLinear(layers) => hybrid_cache::reserve(layers, tokens),
        }
    }

    pub fn cached_tokens(&self) -> Result<usize> {
        match &self.storage {
            CacheStorage::Attention(caches) => {
                caches.first().map_or_else(|| Ok(0), KvCache::offset)
            },
            CacheStorage::HybridLinear(layers) => hybrid_cache::offset(layers),
        }
    }

    pub fn snapshot_at(&self, offset: usize) -> Result<Self> {
        match &self.storage {
            CacheStorage::Attention(caches) => {
                let caches = caches
                    .iter()
                    .map(|cache| cache.snapshot_at(offset))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self { storage: CacheStorage::Attention(caches) })
            },
            CacheStorage::HybridLinear(layers) => {
                let layers = hybrid_cache::snapshot_at(layers, offset)?;
                Ok(Self {
                    storage: CacheStorage::HybridLinear(layers),
                })
            },
        }
    }
}
