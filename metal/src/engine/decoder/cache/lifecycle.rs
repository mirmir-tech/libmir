use super::{CacheStorage, DecoderCache, hybrid};
use crate::engine::{Error, Result, Stream};

impl DecoderCache {
    pub(crate) fn state_space_state(
        &mut self,
        index: usize,
    ) -> Result<&mut crate::engine::state_space_hybrid::StateSpaceState> {
        match &mut self.storage {
            CacheStorage::HybridLinear(layers) => hybrid::state_space_state(layers, index),
            CacheStorage::Attention(_) => {
                Err(Error::InvalidModel("expected hybrid linear cache".into()))
            },
        }
    }

    pub(crate) fn release_history(&mut self) {
        match &mut self.storage {
            CacheStorage::Attention(caches) => {
                for cache in caches {
                    cache.release_history();
                }
            },
            CacheStorage::HybridLinear(layers) => {
                for layer in layers {
                    if let hybrid::HybridLinearLayerCache::Full(cache) = layer {
                        cache.release_history();
                    }
                }
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn history_allocations(
        &self,
    ) -> Result<std::collections::HashSet<mirtal::memory::Allocation>> {
        let mut allocations = std::collections::HashSet::new();
        match &self.storage {
            CacheStorage::Attention(caches) => {
                for cache in caches {
                    cache.extend_history_allocations(&mut allocations)?;
                }
            },
            CacheStorage::HybridLinear(layers) => {
                for layer in layers {
                    if let hybrid::HybridLinearLayerCache::Full(cache) = layer {
                        cache.extend_history_allocations(&mut allocations)?;
                    }
                }
            },
        }
        Ok(allocations)
    }

    #[cfg(test)]
    pub(crate) fn prepare_prefix_retention(&mut self, stream: &Stream) -> Result<()> {
        if stream.config().diagnostics.prefix_retention == crate::config::PrefixRetention::View {
            return Ok(());
        }
        if let CacheStorage::HybridLinear(layers) = &mut self.storage {
            for layer in layers {
                if let hybrid::HybridLinearLayerCache::Linear(state) = layer {
                    state.prepare_prefix_retention(stream)?;
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn convolution_allocations(
        &self,
    ) -> Result<std::collections::HashSet<mirtal::memory::Allocation>> {
        let mut allocations = std::collections::HashSet::new();
        if let CacheStorage::HybridLinear(layers) = &self.storage {
            for layer in layers {
                if let hybrid::HybridLinearLayerCache::Linear(state) = layer
                    && let Some(root) = state.convolution_root()
                {
                    allocations.insert(root.native().allocation()?.ok_or_else(|| {
                        Error::InvalidModel(
                            "convolution inventory requires materialized state".into(),
                        )
                    })?);
                }
            }
        }
        Ok(allocations)
    }

    pub(crate) fn extend_graph_roots<'a>(&'a self, roots: &mut Vec<&'a crate::engine::Array>) {
        if let CacheStorage::HybridLinear(layers) = &self.storage {
            roots.extend(hybrid::graph_roots(layers));
        }
    }

    /// Includes contiguous and sliding K/V, which a batched history no longer
    /// reads.
    pub(crate) fn extend_prefill_graph_roots<'a>(
        &'a self,
        roots: &mut Vec<&'a crate::engine::Array>,
    ) {
        self.extend_graph_roots(roots);
        match &self.storage {
            CacheStorage::Attention(caches) => {
                for cache in caches {
                    cache.extend_graph_roots(roots);
                }
            },
            CacheStorage::HybridLinear(layers) => {
                for layer in layers {
                    if let hybrid::HybridLinearLayerCache::Full(cache) = layer {
                        cache.extend_graph_roots(roots);
                    }
                }
            },
        }
    }

    pub(crate) fn recurrent_allocations(
        &self,
    ) -> Result<std::collections::HashSet<mirtal::memory::Allocation>> {
        let mut roots = Vec::new();
        self.extend_graph_roots(&mut roots);
        roots
            .into_iter()
            .map(|array| {
                array.native().allocation()?.ok_or_else(|| {
                    Error::InvalidModel(
                        "recurrent prefix state must be materialized before accounting".into(),
                    )
                })
            })
            .collect()
    }

    /// Logical bytes of the distinct recurrent buffers: a row view counts
    /// its whole cohort parent. Allocator capacities would add the buffer
    /// cache's rounding, which depends on what else the process freed.
    #[cfg(test)]
    pub(crate) fn recurrent_logical_bytes(&self) -> Result<usize> {
        let mut roots = Vec::new();
        self.extend_graph_roots(&mut roots);
        let mut seen = std::collections::HashSet::new();
        roots.into_iter().try_fold(0, |total, root| {
            let native = root.native();
            let allocation = native.allocation()?.ok_or_else(|| {
                Error::InvalidModel("recurrent state must be materialized".into())
            })?;
            if !seen.insert(allocation) {
                return Ok(total);
            }
            let width = match native.dtype()? {
                mirtal::DType::Bool | mirtal::DType::Uint8 => 1,
                mirtal::DType::Float16 | mirtal::DType::Bfloat16 => 2,
                mirtal::DType::Uint32 | mirtal::DType::Int32 | mirtal::DType::Float32 => 4,
            };
            Ok(total + native.shape()?.elements()? * width)
        })
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
