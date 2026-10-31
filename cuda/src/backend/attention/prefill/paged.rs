use mircuda::{DeviceBuffer, bf16};

use super::{DecodeAttentionBf16, DecodeAttentionWeights, PrefillAttentionBf16};
use crate::{Error, PagedPrefillBatch, Result};

impl PrefillAttentionBf16 {
    pub(super) fn execute_paged_varlen_attention(
        &mut self,
        state: &DecodeAttentionBf16,
        weights: DecodeAttentionWeights<'_>,
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        // Rows past the batch's tokens are padding of a packed pass.
        let rows = batch.tokens();
        let query = self.scratch.query_rope.slice(0..rows * self.query_width)?;
        let mut attended = self.scratch.attention.slice(0..rows * self.attention_width)?;
        self.fmha
            .as_ref()
            .ok_or(Error::InvalidPagedKv("missing paged variable-length FMHA"))?
            .execute_paged_varlen(
                &self.stream,
                &query,
                state.cache.key_pages(),
                state.cache.value_pages(),
                &mut attended,
                batch.query_starts(),
                batch.token_counts(),
                batch.context_starts(),
                batch.tables(),
                &mut self.scratch.normalized,
                batch.active(),
                batch.tokens(),
                batch.max_query_tokens(),
                batch.max_context_tokens(),
                batch.max_blocks(),
                batch.cache_config().block_size,
                self.config.attention_scale,
            )?;
        self.output_projection.execute(
            &self.stream,
            &self.scratch.attention,
            weights.output,
            output,
        )
    }
}

impl PrefillAttentionBf16 {
    /// Follows a K/V resize of the pages this prefill writes; only the block
    /// count may change.
    pub(in crate::backend) fn follow_cache(
        &mut self,
        storage: runtime::kv::KvStorageSpec,
    ) -> Result<()> {
        let mut expected = self.config.cache;
        expected.cache.block_count = storage.cache.block_count;
        if storage != expected {
            return Err(Error::InvalidPagedKv("resized cache differs from prefill attention"));
        }
        self.config.cache = storage;
        Ok(())
    }
}
