//! Read access to the packed batch: its geometry and device metadata.

use mircuda::DeviceBuffer;
use runtime::kv::CacheConfig;

use super::{PagedPrefillBatch, PrefillBatchRow};

impl PagedPrefillBatch {
    #[must_use]
    pub const fn active(&self) -> usize {
        self.active
    }

    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) const fn slot_mapping(&self, windowed: bool) -> &DeviceBuffer<u32> {
        if windowed {
            &self.ring_slot_mapping.device
        } else {
            &self.slot_mapping.device
        }
    }

    pub(crate) const fn tables(&self) -> &DeviceBuffer<u32> {
        &self.tables.device
    }

    /// Follows a resized K/V cache: block tables may then address up to
    /// `block_count` blocks. Row capacity and page geometry do not change.
    pub(crate) const fn follow_cache_blocks(&mut self, block_count: u32) {
        self.cache.block_count = block_count;
    }

    pub(crate) const fn block_counts(&self) -> &DeviceBuffer<u32> {
        &self.block_counts.device
    }

    pub(crate) const fn query_starts(&self) -> &DeviceBuffer<u32> {
        &self.query_starts.device
    }

    pub(crate) const fn context_starts(&self) -> &DeviceBuffer<u32> {
        &self.context_starts.device
    }

    pub(crate) const fn request_indices(&self) -> &DeviceBuffer<u32> {
        &self.request_indices.device
    }

    pub(crate) const fn positions(&self) -> &DeviceBuffer<u32> {
        &self.positions.device
    }

    pub(crate) fn rows(&self) -> &[PrefillBatchRow] {
        &self.rows
    }

    pub(crate) const fn max_query_tokens(&self) -> usize {
        self.max_query_tokens
    }

    pub(crate) const fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    pub(crate) const fn cache_config(&self) -> CacheConfig {
        self.cache
    }
}
