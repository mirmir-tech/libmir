//! Where on-demand storage takes its blocks. A backend reading a table's
//! pages as one run (Metal) keeps its fast path only while the table's
//! blocks are consecutive, so a table grows right after its last block, or
//! takes the first run of free blocks long enough, before scattered ones.

use super::{BlockId, BlockTable, KvCache, KvResidency};
use crate::error::{Result, RuntimeError};

impl KvCache {
    /// Allocates a block for a table ending at `after`, right after it when
    /// storage grows on demand and that block is free.
    pub fn allocate_after(&mut self, after: Option<BlockId>) -> Result<BlockId> {
        self.ensure_free_blocks(1)?;
        self.reuse_resident_blocks(1)?;
        if let Some(start) = self.placement(1, after) {
            return self.take(BlockId(start));
        }
        let id = self
            .free
            .first()
            .copied()
            .ok_or_else(|| RuntimeError::KvCache("no free KV blocks".into()))?;
        self.take(id)
    }

    /// Allocates `count` blocks for a table ending at `after`, consecutive
    /// where storage grows on demand (`placement`).
    pub(crate) fn allocate_blocks_after(
        &mut self,
        count: usize,
        after: Option<BlockId>,
    ) -> Result<BlockTable> {
        self.ensure_free_blocks(count)?;
        self.reuse_resident_blocks(count)?;
        let mut table = BlockTable::with_block_size(self.config.block_size);
        if let Some(start) = self.placement(count, after) {
            for id in (start..).take(count) {
                table.push(self.take(BlockId(id))?);
            }
            return Ok(table);
        }
        let mut last = after;
        for _ in 0..count {
            let block = self.allocate_after(last)?;
            table.push(block);
            last = Some(block);
        }
        Ok(table)
    }

    /// The first of `count` consecutive free blocks to take for a table
    /// ending at `after`: right after it, else the first run long enough
    /// below the resident blocks, else the resident end. `None` takes the
    /// lowest free blocks.
    pub(super) fn placement(&self, count: usize, after: Option<BlockId>) -> Option<u32> {
        if self.residency != KvResidency::OnDemand || count == 0 {
            return None;
        }
        let count = u32::try_from(count).ok()?;
        let free_run = |start: u32| {
            start.checked_add(count).is_some_and(|end| {
                end <= self.config.block_count
                    && (start..end).all(|id| self.free.contains(&BlockId(id)))
            })
        };
        if let Some(start) =
            after.and_then(|last| last.0.checked_add(1)).filter(|&start| free_run(start))
        {
            return Some(start);
        }
        if let Some(start) = self.run_below_resident(count) {
            return Some(start);
        }
        free_run(self.resident).then_some(self.resident)
    }

    /// The first run of `count` consecutive free blocks below the resident
    /// ones.
    pub(super) fn run_below_resident(&self, count: u32) -> Option<u32> {
        let mut run_start = None;
        let mut previous = None;
        for &BlockId(id) in self.free.range(..BlockId(self.resident)) {
            if previous.and_then(|previous: u32| previous.checked_add(1)) != Some(id) {
                run_start = Some(id);
            }
            previous = Some(id);
            if let Some(start) = run_start
                && id - start + 1 >= count
            {
                return Some(start);
            }
        }
        None
    }

    /// Takes the free block `id`.
    pub(super) fn take(&mut self, id: BlockId) -> Result<BlockId> {
        if !self.free.remove(&id) {
            return Err(RuntimeError::KvCache("KV block is not free".into()));
        }
        self.resident = self.resident.max(id.0.saturating_add(1));
        self.block_mut(id)?.allocate();
        Ok(id)
    }
}
