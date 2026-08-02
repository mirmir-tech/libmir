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
        self.fmha
            .as_ref()
            .ok_or(Error::InvalidPagedKv("missing paged variable-length FMHA"))?
            .execute_paged_varlen(
                &self.stream,
                &self.scratch.query_rope,
                state.cache.key_pages(),
                state.cache.value_pages(),
                &mut self.scratch.attention,
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
