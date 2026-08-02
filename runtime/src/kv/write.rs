use uuid::Uuid;

use super::{BlockId, BlockTable};
use crate::error::{Result, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvPageId {
    pub layer: usize,
    pub block: BlockId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBlockWrite {
    pub page: KvPageId,
    pub table_index: usize,
    pub local_start: usize,
    pub local_end: usize,
    pub page_start: usize,
    pub page_end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvWritePlan {
    block_size: usize,
    token_count: usize,
    writes: Vec<KvBlockWrite>,
}

impl KvPageId {
    #[must_use]
    pub const fn new(_session_id: Uuid, layer: usize, block: BlockId) -> Self {
        Self { layer, block }
    }
}

impl KvBlockWrite {
    #[must_use]
    pub const fn token_count(self) -> usize {
        self.local_end - self.local_start
    }

    #[must_use]
    pub const fn page_token_count(self) -> usize {
        self.page_end - self.page_start
    }
}

impl KvWritePlan {
    pub fn prefill(
        session_id: Uuid,
        layer: usize,
        table: &BlockTable,
        token_offset: usize,
        token_count: usize,
    ) -> Result<Self> {
        let block_size = table
            .block_size()
            .ok_or_else(|| RuntimeError::KvCache("KV block table has no block size".into()))?;
        let mut writes = Vec::new();
        for (index, block) in table.blocks().iter().copied().enumerate() {
            let block_start = index * block_size;
            let block_end = block_start + block_size;
            let token_end = token_offset.saturating_add(token_count);
            let start = block_start.max(token_offset);
            let end = block_end.min(token_end);
            if start < end {
                writes.push(KvBlockWrite {
                    page: KvPageId::new(session_id, layer, block),
                    table_index: index,
                    local_start: start - token_offset,
                    local_end: end - token_offset,
                    page_start: start - block_start,
                    page_end: end - block_start,
                });
            }
        }
        Ok(Self { block_size, token_count, writes })
    }

    #[must_use]
    pub fn writes(&self) -> &[KvBlockWrite] {
        &self.writes
    }

    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.token_count
    }

    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    #[must_use]
    pub fn written_tokens(&self) -> usize {
        self.writes.iter().map(|write| write.token_count()).sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }

    /// Drops writes for a local prefix while preserving tensor-relative
    /// offsets for the remaining suffix.
    pub fn skip_prefix(&mut self, token_count: usize) {
        let skipped = token_count.min(self.token_count);
        self.writes.retain_mut(|write| {
            if write.local_end <= skipped {
                return false;
            }
            if write.local_start < skipped {
                let delta = skipped - write.local_start;
                write.local_start = skipped;
                write.page_start += delta;
            }
            true
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_prefill_writes_across_blocks() -> Result<()> {
        let session_id = Uuid::new_v4();
        let mut table = BlockTable::with_block_size(2);
        table.push(BlockId(7));
        table.push(BlockId(8));

        let plan = KvWritePlan::prefill(session_id, 3, &table, 1, 3)?;

        assert_eq!(plan.token_count(), 3);
        assert_eq!(plan.block_size(), 2);
        assert_eq!(plan.written_tokens(), 3);
        assert_eq!(
            plan.writes(),
            &[
                KvBlockWrite {
                    page: KvPageId::new(session_id, 3, BlockId(7)),
                    table_index: 0,
                    local_start: 0,
                    local_end: 1,
                    page_start: 1,
                    page_end: 2,
                },
                KvBlockWrite {
                    page: KvPageId::new(session_id, 3, BlockId(8)),
                    table_index: 1,
                    local_start: 1,
                    local_end: 3,
                    page_start: 0,
                    page_end: 2,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn skips_cached_prefix_without_rebasing_tensor_offsets() -> Result<()> {
        let session_id = Uuid::new_v4();
        let mut table = BlockTable::with_block_size(2);
        table.push(BlockId(7));
        table.push(BlockId(8));
        let mut plan = KvWritePlan::prefill(session_id, 3, &table, 0, 4)?;

        plan.skip_prefix(3);

        assert_eq!(
            plan.writes(),
            &[KvBlockWrite {
                page: KvPageId::new(session_id, 3, BlockId(8)),
                table_index: 1,
                local_start: 3,
                local_end: 4,
                page_start: 1,
                page_end: 2,
            }]
        );
        Ok(())
    }
}
