mod capability;
mod config;
mod experts;
mod layer;
mod layout;
mod load;
mod plan;
mod projection;
mod scratch;
mod session;
mod validation;
mod weights;

use architecture::CacheLayout;
use models::layout::DecoderConfig;
use runtime::kv::{CacheConfig, ring_blocks};

pub use self::session::CudaClampedRoutedModelSession;
use self::{
    capability::{ClampedRoutedCapabilityPlan, ClampedRoutedQkvLowering},
    config::ClampedRoutedConfig,
    layer::ClampedRoutedLayerTemplate,
    layout::ClampedRoutedLayout,
    projection::{ClampedRoutedBoundaryProjection, ClampedRoutedOutputProjection},
};
use crate::{CudaBackend, CudaTensor, PagedKvCache, Result};

const PREFIX_CHECKPOINTS_PER_SESSION: usize = 3;
const WINDOWED_FMHA_MIN_QUERY_TOKENS: usize = 128;

#[derive(Clone)]
pub struct CudaClampedRoutedModelTemplate {
    backend: CudaBackend,
    decoder: DecoderConfig,
    cache_layout: CacheLayout,
    embedding: ClampedRoutedBoundaryProjection,
    final_norm: CudaTensor,
    output: ClampedRoutedOutputProjection,
    layers: Vec<ClampedRoutedLayerTemplate>,
    config: ClampedRoutedConfig,
    cache: CacheConfig,
    max_sequence_blocks: usize,
    ring_sessions: usize,
}

impl CudaClampedRoutedModelTemplate {
    #[must_use]
    pub const fn decoder(&self) -> &DecoderConfig {
        &self.decoder
    }

    pub fn instantiate(&self) -> Result<CudaClampedRoutedModelSession> {
        let caches = self.allocate_shared_kv()?;
        self.instantiate_with_caches(&caches)
    }

    pub(crate) fn allocate_shared_kv(&self) -> Result<Vec<PagedKvCache>> {
        self.backend
            .allocate_cache_layout(&self.cache_layout, self.cache, self.ring_sessions, false)?
            .into_iter()
            .map(|cache| {
                cache.ok_or(crate::Error::InvalidPagedKv(
                    "clamped-routed layer requires a paged K/V cache",
                ))
            })
            .collect()
    }

    #[must_use]
    pub(crate) fn max_sliding_window(&self) -> Option<usize> {
        self.cache_layout.max_ring_window()
    }

    pub(super) fn ring_blocks(&self) -> Option<usize> {
        self.max_sliding_window()
            .map(|window| ring_blocks(window, self.cache.block_size))
    }

    pub(super) fn checkpoint_slots(&self) -> Result<usize> {
        self.ring_sessions
            .checked_mul(PREFIX_CHECKPOINTS_PER_SESSION)
            .ok_or(crate::Error::InvalidPagedKv("windowed KV checkpoint capacity overflow"))
    }

    pub(crate) fn prefix_replay_tokens(&self) -> Option<usize> {
        let mut windows = self.layers.iter().filter_map(ClampedRoutedLayerTemplate::window);
        let first = windows.next()?;
        Some(windows.fold(first, |replay, window| replay.saturating_add(window.saturating_sub(1))))
    }

    pub(crate) const fn prefix_checkpoint_block_tokens(&self) -> usize {
        self.cache.block_size
    }

    pub(crate) fn instantiate_with_caches(
        &self,
        caches: &[PagedKvCache],
    ) -> Result<CudaClampedRoutedModelSession> {
        CudaClampedRoutedModelSession::new(self, caches)
    }
}
