use std::sync::Arc;

use self::hybrid::HybridLinearLayerCache;
use crate::engine::{
    Error, GatedDeltaState, KvCache, KvPageFormat, PagedArenaPool, Result, Stream,
    lowering::MixerLowering,
};

mod hybrid;
#[cfg(test)]
mod tests;

const TOKEN_ROUNDING_PAGES_PER_SESSION: usize = 1;
const CONTIGUOUS_TAIL_PAGES_PER_SESSION: usize = 1;
const COPY_ON_WRITE_PAGES_PER_SESSION: usize = 1;

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
    pub(crate) fn physical_page_capacity(stream: &Stream, allocation_step_tokens: usize) -> usize {
        let page_size = stream.config().kv_cache.block_size.max(1);
        let allocation_step_pages = allocation_step_tokens.div_ceil(page_size).max(1);
        physical_page_capacity(
            stream.config().kv_cache.block_count as usize,
            stream.config().max_batch_requests(),
            allocation_step_pages,
        )
    }

    pub fn new(cache_windows: &[Option<usize>], step: usize) -> Result<Self> {
        Self::new_with_format(cache_windows, step, KvPageFormat::Native, 16)
    }

    pub(crate) fn new_with_format(
        cache_windows: &[Option<usize>],
        step: usize,
        format: KvPageFormat,
        page_size: usize,
    ) -> Result<Self> {
        Self::new_with_pool(
            cache_windows,
            step,
            format,
            page_size,
            &Arc::new(PagedArenaPool::default()),
        )
    }

    pub(crate) fn new_with_pool(
        cache_windows: &[Option<usize>],
        step: usize,
        format: KvPageFormat,
        page_size: usize,
        pool: &Arc<PagedArenaPool>,
    ) -> Result<Self> {
        Self::new_with_pool_capacity(cache_windows, step, format, page_size, usize::MAX, pool)
    }

    pub(crate) fn new_with_pool_capacity(
        cache_windows: &[Option<usize>],
        step: usize,
        format: KvPageFormat,
        page_size: usize,
        max_pages: usize,
        pool: &Arc<PagedArenaPool>,
    ) -> Result<Self> {
        let caches = cache_windows
            .iter()
            .enumerate()
            .map(|(layer, window)| {
                if window.is_none() {
                    KvCache::new_paged_with_pool_capacity(
                        step,
                        page_size,
                        format,
                        max_pages,
                        Arc::clone(pool),
                        layer,
                    )
                } else {
                    KvCache::new_with_window(step, *window)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { storage: CacheStorage::Attention(caches) })
    }

    pub fn new_hybrid_linear(mixers: &[MixerLowering], step: usize) -> Result<Self> {
        Self::new_hybrid_linear_with_format(mixers, step, KvPageFormat::Native, 16)
    }

    pub(crate) fn new_hybrid_linear_with_format(
        mixers: &[MixerLowering],
        step: usize,
        format: KvPageFormat,
        page_size: usize,
    ) -> Result<Self> {
        Self::new_hybrid_linear_with_pool(
            mixers,
            step,
            format,
            page_size,
            &Arc::new(PagedArenaPool::default()),
        )
    }

    pub(crate) fn new_hybrid_linear_with_pool(
        mixers: &[MixerLowering],
        step: usize,
        format: KvPageFormat,
        page_size: usize,
        pool: &Arc<PagedArenaPool>,
    ) -> Result<Self> {
        Self::new_hybrid_linear_with_pool_capacity(
            mixers,
            step,
            format,
            page_size,
            usize::MAX,
            pool,
        )
    }

    pub(crate) fn new_hybrid_linear_with_pool_capacity(
        mixers: &[MixerLowering],
        step: usize,
        format: KvPageFormat,
        page_size: usize,
        max_pages: usize,
        pool: &Arc<PagedArenaPool>,
    ) -> Result<Self> {
        let layers = hybrid::new(mixers, step, format, page_size, max_pages, pool)?;
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
            CacheStorage::HybridLinear(layers) => hybrid::gated_delta_state(layers, index),
            CacheStorage::Attention(_) => {
                Err(Error::InvalidModel("expected hybrid linear cache".into()))
            },
        }
    }

    pub(crate) fn full_attention_cache(&mut self, index: usize) -> Result<&mut KvCache> {
        match &mut self.storage {
            CacheStorage::HybridLinear(layers) => hybrid::full_attention_cache(layers, index),
            CacheStorage::Attention(_) => {
                Err(Error::InvalidModel("expected hybrid linear cache".into()))
            },
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        match &mut self.storage {
            CacheStorage::Attention(caches) => caches.iter_mut().try_for_each(KvCache::reset),
            CacheStorage::HybridLinear(layers) => hybrid::reset(layers),
        }
    }

    pub fn reserve(&mut self, tokens: usize) -> Result<()> {
        match &mut self.storage {
            CacheStorage::Attention(caches) => {
                caches.iter_mut().try_for_each(|cache| cache.reserve(tokens))
            },
            CacheStorage::HybridLinear(layers) => hybrid::reserve(layers, tokens),
        }
    }

    pub(crate) fn plan_contiguous(&mut self, tokens: usize) {
        match &mut self.storage {
            CacheStorage::Attention(caches) => {
                for cache in caches {
                    cache.plan_contiguous(tokens);
                }
            },
            CacheStorage::HybridLinear(layers) => hybrid::plan_contiguous(layers, tokens),
        }
    }

    pub(crate) fn detach_evaluated_graphs(&self) -> Result<()> {
        match &self.storage {
            CacheStorage::Attention(caches) => {
                caches.iter().try_for_each(KvCache::detach_evaluated_graphs)
            },
            CacheStorage::HybridLinear(layers) => hybrid::detach_evaluated_graphs(layers),
        }
    }

    pub fn cached_tokens(&self) -> Result<usize> {
        match &self.storage {
            CacheStorage::Attention(caches) => {
                caches.first().map_or_else(|| Ok(0), KvCache::offset)
            },
            CacheStorage::HybridLinear(layers) => hybrid::offset(layers),
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
                let layers = hybrid::snapshot_at(layers, offset)?;
                Ok(Self {
                    storage: CacheStorage::HybridLinear(layers),
                })
            },
        }
    }

    pub(crate) fn supports_prefix_offsets(&self) -> bool {
        match &self.storage {
            CacheStorage::Attention(caches) => caches.iter().all(KvCache::supports_prefix_offsets),
            CacheStorage::HybridLinear(_) => false,
        }
    }
}

fn physical_page_capacity(logical: usize, sessions: usize, allocation_step: usize) -> usize {
    let per_session = TOKEN_ROUNDING_PAGES_PER_SESSION
        + CONTIGUOUS_TAIL_PAGES_PER_SESSION
        + COPY_ON_WRITE_PAGES_PER_SESSION;
    logical
        .saturating_add(sessions.max(1).saturating_mul(per_session))
        .div_ceil(allocation_step.max(1))
        .saturating_mul(allocation_step.max(1))
}
