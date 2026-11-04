use super::{BatchedPagedAttentionBf16, fmha::PagedFmhaDecode};
use crate::PagedDecodeBatch;

impl BatchedPagedAttentionBf16 {
    /// What a captured graph of `execute` with `window` depends on: the
    /// `FlashAttention` context band, or the split partitions it launches.
    pub(crate) fn capture_partitions(
        &self,
        batch: &PagedDecodeBatch,
        window: Option<usize>,
    ) -> usize {
        if window.is_none() && self.fmha.is_some() {
            return PagedFmhaDecode::capture_key(batch);
        }
        let tokens = batch.maximum_tokens();
        let visible = window.map_or(tokens, |window| tokens.min(window));
        visible.div_ceil(self.split.partition_tokens()).min(self.split.max_partitions())
    }
}
