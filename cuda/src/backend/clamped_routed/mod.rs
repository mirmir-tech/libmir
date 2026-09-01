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

use models::layout::DecoderConfig;
use runtime::kv::CacheConfig;

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
        let storage = self.config.storage(self.cache);
        let ring_window = self.max_sliding_window();
        let ring_slots = self.ring_slots()?;
        self.layers
            .iter()
            .enumerate()
            .map(|(layer, template)| match template.window() {
                Some(_) => self.backend.prepare_windowed_paged_kv(
                    layer,
                    storage,
                    ring_window.unwrap_or(self.cache.block_size),
                    ring_slots,
                ),
                None => self.backend.prepare_paged_kv(layer, storage),
            })
            .collect()
    }

    #[must_use]
    pub(crate) fn max_sliding_window(&self) -> Option<usize> {
        self.layers.iter().filter_map(ClampedRoutedLayerTemplate::window).max()
    }

    pub(super) fn ring_blocks(&self) -> Option<usize> {
        self.max_sliding_window().map(|window| {
            window
                .saturating_add(self.cache.block_size.saturating_sub(1))
                .div_ceil(self.cache.block_size)
        })
    }

    pub(super) fn checkpoint_slots(&self) -> Result<usize> {
        self.ring_sessions
            .checked_mul(PREFIX_CHECKPOINTS_PER_SESSION)
            .ok_or(crate::Error::InvalidPagedKv("windowed KV checkpoint capacity overflow"))
    }

    fn ring_slots(&self) -> Result<usize> {
        self.ring_sessions
            .checked_add(self.checkpoint_slots()?)
            .ok_or(crate::Error::InvalidPagedKv("windowed KV slot capacity overflow"))
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
