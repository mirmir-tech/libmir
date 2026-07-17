use uuid::Uuid;

use super::{BlockHash, BlockId, BlockTable, KvCache, KvWritePlan};
use crate::error::{Result, RuntimeError};

#[derive(Debug, Clone)]
pub struct KvSessionState {
    session_id: Uuid,
    model: String,
    table: BlockTable,
    tokens: Vec<u32>,
    committed_blocks: usize,
    last_hash: Option<BlockHash>,
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
        }
    }

    pub fn prepare_prefill(
        &mut self,
        cache: &mut KvCache,
        prompt_tokens: &[u32],
    ) -> Result<KvPrefillReservation> {
        let step = self.prepare_prefill_in_place(cache, prompt_tokens)?;
        Ok(KvPrefillReservation {
            session_id: step.session_id,
            table: self.table.clone(),
            cached_tokens: step.cached_tokens,
            missing_tokens: step.missing_tokens,
            write_offset: step.write_offset,
        })
    }

    pub fn prepare_prefill_in_place(
        &mut self,
        cache: &mut KvCache,
        prompt_tokens: &[u32],
    ) -> Result<KvPrefillStep> {
        if !self.table.is_empty() {
            self.release(cache)?;
        }
        let probe = cache.probe_prefix_recorded(&self.model, prompt_tokens);
        let missing_blocks = probe.missing_tokens.div_ceil(cache.block_size());
        let allocated = cache.allocate_blocks(missing_blocks)?;
        let mut table = BlockTable::with_block_size(cache.block_size());
        for block in probe.cached_blocks.iter().copied() {
            cache.retain(block)?;
            table.push(block);
        }
        for block in allocated.blocks().iter().copied() {
            table.push(block);
        }
        table.set_token_len(prompt_tokens.len());
        self.table = table;
        self.tokens = prompt_tokens.to_vec();
        self.committed_blocks = probe.cached_blocks.len();
        self.last_hash = probe.last_hash;
        Ok(KvPrefillStep {
            session_id: self.session_id,
            cached_tokens: probe.cached_tokens,
            missing_tokens: probe.missing_tokens,
            write_offset: probe.cached_tokens,
        })
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

    pub fn commit_ready_prefix_blocks(&mut self, cache: &mut KvCache) -> Result<usize> {
        let block_size = cache.block_size();
        let mut committed = 0;
        while self.committed_blocks < self.table.blocks().len() {
            let start = self.committed_blocks * block_size;
            let end = (start + block_size).min(self.tokens.len());
            if end - start < block_size {
                break;
            }
            let block = self.table.blocks()[self.committed_blocks];
            let hash = cache.commit_prefix_block(
                &self.model,
                self.last_hash,
                block,
                &self.tokens[start..end],
            )?;
            self.last_hash = Some(hash);
            self.committed_blocks += 1;
            committed += 1;
        }
        Ok(committed)
    }

    pub fn release(&mut self, cache: &mut KvCache) -> Result<usize> {
        let released = cache.release_table(&self.table)?;
        self.table = BlockTable::with_block_size(cache.block_size());
        self.tokens.clear();
        self.committed_blocks = 0;
        self.last_hash = None;
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
