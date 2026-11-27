//! The cache a layer appends its keys and values to. A layer attending over
//! an earlier layer's keys and values (Gemma 4's E models) keeps none: the
//! model allocates caches for its other layers only, which precede it.

use super::HybridMoeLayer;
use crate::engine::{DecoderCache, Error, KvCache, Result, hybrid::moe::weights::KeyValueWeights};

impl HybridMoeLayer {
    /// Whether the layer caches the keys and values it projects.
    pub(in crate::engine::hybrid::moe) const fn caches_key_values(&self) -> bool {
        matches!(self.weights.attention.key_values, KeyValueWeights::Projected(_))
    }

    /// Layer `index`'s cache in `cache`; `None` for a sharing layer.
    pub(in crate::engine::hybrid::moe) fn own_cache<'a>(
        &self,
        cache: &'a mut DecoderCache,
        index: usize,
    ) -> Result<Option<&'a mut KvCache>> {
        if !self.caches_key_values() {
            return Ok(None);
        }
        cache
            .attention_caches_mut()?
            .get_mut(index)
            .map(Some)
            .ok_or_else(|| Error::InvalidModel(format!("missing cache for layer {index}")))
    }

    /// Layer `index`'s cache of every packed row; none for a sharing layer.
    pub(in crate::engine::hybrid::moe) fn own_caches<'a>(
        &self,
        caches: &'a mut [&mut DecoderCache],
        index: usize,
    ) -> Result<Vec<&'a mut KvCache>> {
        caches
            .iter_mut()
            .filter_map(|cache| self.own_cache(cache, index).transpose())
            .collect()
    }
}
