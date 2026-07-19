use super::{KvPrefillStep, KvSessionState};
use crate::{
    error::Result,
    kv::{BlockTable, KvCache},
};

impl KvSessionState {
    /// Allocates a fresh prefill table and prevents this session from
    /// publishing or reusing token-only prefix-cache entries.
    pub fn prepare_uncached_prefill_in_place(
        &mut self,
        cache: &mut KvCache,
        prompt_tokens: &[u32],
    ) -> Result<KvPrefillStep> {
        if !self.table.is_empty() {
            self.release(cache)?;
        }
        let block_count = prompt_tokens.len().div_ceil(cache.block_size());
        let allocated = cache.allocate_blocks(block_count)?;
        let mut table = BlockTable::with_block_size(cache.block_size());
        for block in allocated.blocks().iter().copied() {
            table.push(block);
        }
        table.set_token_len(prompt_tokens.len());
        self.table = table;
        self.tokens = prompt_tokens.to_vec();
        self.committed_blocks = 0;
        self.last_hash = None;
        self.prefix_cacheable = false;
        Ok(KvPrefillStep {
            session_id: self.session_id,
            cached_tokens: 0,
            missing_tokens: prompt_tokens.len(),
            write_offset: 0,
        })
    }
}
