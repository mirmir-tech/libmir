use uuid::Uuid;

use super::{BlockHash, BlockId, BlockTable, KvCache, KvWritePlan};
use crate::error::{Result, RuntimeError};

mod prefill;
mod uncached;

#[derive(Debug, Clone)]
pub struct KvSessionState {
    session_id: Uuid,
    model: String,
    table: BlockTable,
    tokens: Vec<u32>,
    committed_blocks: usize,
    last_hash: Option<BlockHash>,
    prefix_cacheable: bool,
}

#[derive(Debug, Clone)]
pub struct KvPrefillReservation {
    pub session_id: Uuid,
    pub table: BlockTable,
    pub cached_tokens: usize,
    pub missing_tokens: usize,
    pub write_offset: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct KvPrefillStep {
    pub session_id: Uuid,
    pub cached_tokens: usize,
    pub missing_tokens: usize,
    pub write_offset: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct KvPrefillPlan {
    pub session_id: Uuid,
    pub cached_tokens: usize,
    pub missing_tokens: usize,
    pub write_offset: usize,
    pub capacity_blocks: usize,
    pub needs_eviction: bool,
}

#[derive(Debug, Clone)]
pub struct KvDecodeReservation {
    pub session_id: Uuid,
    pub table: BlockTable,
    pub token_offset: usize,
    pub allocated_block: Option<BlockId>,
}

#[derive(Debug, Clone, Copy)]
pub struct KvDecodeStep {
    pub session_id: Uuid,
    pub token_offset: usize,
    pub allocated_block: Option<BlockId>,
}

impl KvSessionState {
    #[must_use]
    pub fn new(session_id: Uuid, model: impl Into<String>, block_size: usize) -> Self {
        Self {
            session_id,
            model: model.into(),
            table: BlockTable::with_block_size(block_size),
            tokens: Vec::new(),
            committed_blocks: 0,
            last_hash: None,
            prefix_cacheable: true,
        }
    }

    pub fn append_decode(
        &mut self,
        cache: &mut KvCache,
        token: u32,
    ) -> Result<KvDecodeReservation> {
        let step = self.append_decode_in_place(cache, token)?;
        Ok(KvDecodeReservation {
            session_id: step.session_id,
            table: self.table.clone(),
            token_offset: step.token_offset,
            allocated_block: step.allocated_block,
        })
    }

    pub fn append_decode_in_place(
        &mut self,
        cache: &mut KvCache,
        token: u32,
    ) -> Result<KvDecodeStep> {
        let token_offset = self.tokens.len();
        let allocated_block = if token_offset == self.table.capacity(cache.block_size()) {
            let block = cache.allocate()?;
            self.table.push(block);
            Some(block)
        } else {
            None
        };
        self.tokens.push(token);
        self.table.set_token_len(self.tokens.len());
        Ok(KvDecodeStep {
            session_id: self.session_id,
            token_offset,
            allocated_block,
        })
    }

    pub fn reserve_decode_in_place(&mut self, cache: &mut KvCache) -> Result<KvDecodeStep> {
        self.append_decode_in_place(cache, 0)
    }

    pub fn replace_token_at(&mut self, offset: usize, token: u32) -> Result<()> {
        let len = self.tokens.len();
        let slot = self.tokens.get_mut(offset).ok_or_else(|| {
            RuntimeError::KvCache(format!(
                "cannot replace token at offset {offset}; session has {len} tokens"
            ))
        })?;
        *slot = token;
        Ok(())
    }

    pub fn release(&mut self, cache: &mut KvCache) -> Result<usize> {
        let released = cache.release_table(&self.table)?;
        self.table = BlockTable::with_block_size(cache.block_size());
        self.tokens.clear();
        self.committed_blocks = 0;
        self.last_hash = None;
        self.prefix_cacheable = true;
        Ok(released)
    }

    #[must_use]
    pub fn table(&self) -> &BlockTable {
        &self.table
    }

    #[must_use]
    pub fn token_len(&self) -> usize {
        self.tokens.len()
    }

    #[must_use]
    pub const fn session_id(&self) -> Uuid {
        self.session_id
    }
}

impl KvPrefillReservation {
    pub fn write_plan(&self, layer: usize) -> Result<KvWritePlan> {
        KvWritePlan::prefill(
            self.session_id,
            layer,
            &self.table,
            self.write_offset,
            self.missing_tokens,
        )
    }
}

impl KvDecodeReservation {
    pub fn write_plan(&self, layer: usize) -> Result<KvWritePlan> {
        KvWritePlan::prefill(self.session_id, layer, &self.table, self.token_offset, 1)
    }
}

#[cfg(test)]
mod tests;
