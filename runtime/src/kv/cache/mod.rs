use std::collections::VecDeque;

use super::{BlockHash, BlockId, BlockTable, KvBlock, PrefixCache};
use crate::error::{Result, RuntimeError};

mod pressure;
mod probe;
#[cfg(test)]
mod tests;
mod types;

pub use types::{BlockAllocation, CacheConfig, CacheCounters, CacheStats};

#[derive(Debug)]
pub struct KvCache {
    config: CacheConfig,
    blocks: Vec<KvBlock>,
    free: VecDeque<BlockId>,
    prefix: PrefixCache,
    counters: CacheCounters,
}

impl KvCache {
    #[must_use]
    pub fn new(blocks: u32) -> Self {
        Self::with_config(CacheConfig::new(blocks))
    }

    #[must_use]
    pub fn with_config(config: CacheConfig) -> Self {
        let config = CacheConfig {
            block_size: config.block_size.max(1),
            block_count: config.block_count,
            dtype: config.dtype,
        };
        let mut cache_blocks = Vec::with_capacity(config.block_count as usize);
        let mut free = VecDeque::with_capacity(config.block_count as usize);
        for id in 0..config.block_count {
            let block_id = BlockId(id);
            cache_blocks.push(KvBlock::new(block_id));
            free.push_back(block_id);
        }
        Self {
            config,
            blocks: cache_blocks,
            free,
            prefix: PrefixCache::default(),
            counters: CacheCounters::default(),
        }
    }

    pub fn allocate_for_tokens(&mut self, tokens: usize) -> Result<BlockAllocation> {
        let needed = tokens.div_ceil(self.config.block_size);
        let mut blocks = self.allocate_blocks(needed)?;
        blocks.set_token_len(tokens);
        let token_capacity = blocks.capacity(self.config.block_size);
        Ok(BlockAllocation { table: blocks, token_capacity })
    }

    pub fn allocate(&mut self) -> Result<BlockId> {
        self.ensure_free_blocks(1)?;
        let id = self
            .free
            .pop_front()
            .ok_or_else(|| RuntimeError::KvCache("no free KV blocks".into()))?;
        let block = self.block_mut(id)?;
        block.allocate();
        Ok(id)
    }

    pub fn commit_prefix_block(
        &mut self,
        model: &str,
        parent: Option<BlockHash>,
        block: BlockId,
        tokens: &[u32],
    ) -> Result<BlockHash> {
        let hash = BlockHash::from_tokens(model, parent, tokens);
        let old_hash = {
            let entry = self.block_mut(block)?;
            if entry.is_free() {
                return Err(RuntimeError::KvCache("cannot cache a free KV block".into()));
            }
            let old_hash = entry.hash;
            if old_hash != Some(hash.0) {
                entry.ref_count = entry
                    .ref_count
                    .checked_add(1)
                    .ok_or_else(|| RuntimeError::KvCache("KV block refcount overflow".into()))?;
            }
            entry.hash = Some(hash.0);
            entry.token_count = tokens.len();
            old_hash
        };
        if let Some(old_hash) = old_hash
            && old_hash != hash.0
        {
            let _removed = self.prefix.remove(old_hash);
            let _released = self.release(block)?;
        }
        if let Some(replaced) = self.prefix.insert(hash, block)
            && replaced != block
        {
            self.block_mut(replaced)?.hash = None;
            let _released = self.release(replaced)?;
        }
        Ok(hash)
    }

    pub fn free(&mut self, id: BlockId) -> Result<()> {
        let _released = self.release(id)?;
        Ok(())
    }

    pub fn retain(&mut self, id: BlockId) -> Result<()> {
        let block = self.block_mut(id)?;
        if block.is_free() {
            return Err(RuntimeError::KvCache("cannot retain a free KV block".into()));
        }
        block.ref_count = block
            .ref_count
            .checked_add(1)
            .ok_or_else(|| RuntimeError::KvCache("KV block refcount overflow".into()))?;
        Ok(())
    }

    pub fn release(&mut self, id: BlockId) -> Result<bool> {
        let hash = {
            let block = self.block_mut(id)?;
            if block.is_free() {
                return Err(RuntimeError::KvCache("KV block already free".into()));
            }
            if block.ref_count > 1 {
                block.ref_count -= 1;
                return Ok(false);
            }
            let hash = block.hash;
            block.reset();
            hash
        };
        if let Some(hash) = hash {
            let _removed = self.prefix.remove(hash);
        }
        self.free.push_back(id);
        Ok(true)
    }

    pub fn evict_prefix(&mut self, hash: BlockHash) -> Result<bool> {
        let Some(block) = self.prefix.remove(hash.0) else {
            return Ok(false);
        };
        let released = self.release(block)?;
        self.counters.evictions += 1;
        Ok(released)
    }

    pub fn release_table(&mut self, table: &BlockTable) -> Result<usize> {
        let mut released = 0;
        for block in table.blocks().iter().copied() {
            if self.release(block)? {
                released += 1;
            }
        }
        Ok(released)
    }

    pub fn block_ref_count(&self, id: BlockId) -> Result<u32> {
        Ok(self.block(id)?.ref_count)
    }

    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let total_blocks = self.blocks.len();
        let free_blocks = self.free.len();
        CacheStats {
            block_size: self.config.block_size,
            dtype: self.config.dtype,
            quant_mode: self.config.dtype.quant_mode(),
            total_blocks,
            free_blocks,
            used_blocks: total_blocks.saturating_sub(free_blocks),
            cached_prefixes: self.prefix.len(),
            counters: self.counters,
        }
    }

    #[must_use]
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.config.block_size
    }

    pub(crate) fn allocate_blocks(&mut self, count: usize) -> Result<BlockTable> {
        self.ensure_free_blocks(count)?;
        let mut table = BlockTable::with_block_size(self.config.block_size);
        for _ in 0..count {
            table.push(self.allocate()?);
        }
        Ok(table)
    }

    fn block_mut(&mut self, id: BlockId) -> Result<&mut KvBlock> {
        self.blocks
            .get_mut(id.0 as usize)
            .ok_or_else(|| RuntimeError::KvCache("unknown KV block".into()))
    }

    fn block(&self, id: BlockId) -> Result<&KvBlock> {
        self.blocks
            .get(id.0 as usize)
            .ok_or_else(|| RuntimeError::KvCache("unknown KV block".into()))
    }
}
