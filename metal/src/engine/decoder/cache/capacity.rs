use super::{CacheStorage, DecoderCache, hybrid::HybridLinearLayerCache};
use crate::engine::{DecodeCapacity, Result};

impl DecoderCache {
    pub(crate) fn plan_decode_capacity(
        &self,
        tokens: usize,
        threshold: usize,
        plan: &mut DecodeCapacity,
    ) -> Result<()> {
        match &self.storage {
            CacheStorage::Attention(caches) => {
                for cache in caches {
                    cache.plan_decode_capacity(tokens, threshold, plan)?;
                }
            },
            CacheStorage::HybridLinear(layers) => {
                for layer in layers {
                    if let HybridLinearLayerCache::Full(cache) = layer {
                        cache.plan_decode_capacity(tokens, threshold, plan)?;
                    }
                }
            },
        }
        Ok(())
    }
}
