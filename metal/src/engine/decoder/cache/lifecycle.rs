use super::{CacheStorage, DecoderCache, hybrid};
use crate::engine::{Result, Stream};

impl DecoderCache {
    pub(crate) fn extend_graph_roots<'a>(&'a self, roots: &mut Vec<&'a crate::engine::Array>) {
        if let CacheStorage::HybridLinear(layers) = &self.storage {
            roots.extend(hybrid::graph_roots(layers));
        }
    }

    pub(crate) fn detach_evaluated_graphs(&self, stream: &Stream) -> Result<()> {
        match &self.storage {
            CacheStorage::Attention(caches) => {
                caches.iter().try_for_each(|cache| cache.detach_evaluated_graphs(stream))
            },
            CacheStorage::HybridLinear(layers) => hybrid::detach_evaluated_graphs(layers, stream),
        }
    }
}
