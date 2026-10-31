use super::BatchedDecodeAttentionBf16;
use crate::{Error, PagedDecodeBatch, PagedKvCache, Result};

impl BatchedDecodeAttentionBf16 {
    /// Binds the sessions' pages after a K/V resize replaced them; only the
    /// block count may differ from the pages the batch was built for.
    pub(in crate::backend) fn bind_cache(&mut self, cache: &PagedKvCache) -> Result<()> {
        let storage = cache.storage_spec();
        let mut expected = self.config.cache;
        expected.cache.block_count = storage.cache.block_count;
        if storage != expected || cache.layer() != self.config.layer {
            return Err(Error::InvalidPagedKv("rebound cache differs from batched attention"));
        }
        self.config.cache = storage;
        self.attention.follow_cache_blocks(storage.cache.block_count);
        self.cache = cache.clone();
        Ok(())
    }

    pub(in crate::backend) fn capture_partitions(&self, batch: &PagedDecodeBatch) -> usize {
        self.attention.capture_partitions(batch)
    }
}
