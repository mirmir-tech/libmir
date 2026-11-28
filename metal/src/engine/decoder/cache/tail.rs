//! A prefix snapshot's paged layers resume from the runtime's blocks plus a
//! copy of their partial last page (`kv::paged::tail`).

use super::{CacheStorage, DecoderCache, hybrid::HybridLinearLayerCache};
use crate::engine::{KvCache, Result, Stream};

impl DecoderCache {
    fn key_value_caches(&self) -> Box<dyn Iterator<Item = &KvCache> + '_> {
        match &self.storage {
            CacheStorage::Attention(caches) => Box::new(caches.iter()),
            CacheStorage::HybridLinear(layers) => {
                Box::new(layers.iter().filter_map(|layer| match layer {
                    HybridLinearLayerCache::Full(cache) => Some(cache),
                    HybridLinearLayerCache::Linear(_) | HybridLinearLayerCache::StateSpace(_) => {
                        None
                    },
                }))
            },
        }
    }

    fn key_value_caches_mut(&mut self) -> Box<dyn Iterator<Item = &mut KvCache> + '_> {
        match &mut self.storage {
            CacheStorage::Attention(caches) => Box::new(caches.iter_mut()),
            CacheStorage::HybridLinear(layers) => {
                Box::new(layers.iter_mut().filter_map(|layer| match layer {
                    HybridLinearLayerCache::Full(cache) => Some(cache),
                    HybridLinearLayerCache::Linear(_) | HybridLinearLayerCache::StateSpace(_) => {
                        None
                    },
                }))
            },
        }
    }

    /// Keeps a prefix snapshot's partial last pages, copied on `stream`
    /// before the request that wrote them decodes into them.
    pub(crate) fn capture_tails(&mut self, stream: &Stream) -> Result<()> {
        self.key_value_caches_mut().try_for_each(|cache| cache.capture_tail(stream))?;
        // One evaluation schedules every layer's copy.
        let roots = self.key_value_caches().flat_map(KvCache::tail_roots).collect::<Vec<_>>();
        stream.eval_many(&roots)
    }

    /// Whether a snapshot of this cache at `tokens` resumes from the
    /// runtime's first `cached` tokens.
    pub(crate) fn restorable_within(&self, tokens: usize, cached: usize) -> bool {
        self.key_value_caches().all(|cache| cache.restorable_within(tokens, cached))
    }

    /// Writes a restored snapshot's partial last pages into the blocks bound
    /// since.
    pub(crate) fn restore_tails(&mut self, stream: &Stream) -> Result<()> {
        self.key_value_caches_mut().try_for_each(|cache| cache.restore_tail(stream))
    }
}
