use super::{
    BlockId, BlockTable, KvCache, KvPrefillPlan, KvPrefillReservation, KvPrefillStep,
    KvSessionState, Result, RuntimeError,
};

impl KvSessionState {
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
        self.prepare_prefill_with_reserve_in_place(cache, prompt_tokens, 0)
    }

    pub fn prepare_prefill_with_reserve_in_place(
        &mut self,
        cache: &mut KvCache,
        prompt_tokens: &[u32],
        reserved_tokens: usize,
    ) -> Result<KvPrefillStep> {
        let plan =
            self.probe_prefill_with_reserve_in_place(cache, prompt_tokens, reserved_tokens)?;
        match self.allocate_prefill_plan_in_place(cache, plan) {
            Ok(step) => Ok(step),
            Err(error) => {
                self.release(cache)?;
                Err(error)
            },
        }
    }

    pub fn probe_prefill_with_reserve_in_place(
        &mut self,
        cache: &mut KvCache,
        prompt_tokens: &[u32],
        reserved_tokens: usize,
    ) -> Result<KvPrefillPlan> {
        if !self.table.is_empty() {
            self.release(cache)?;
        }
        self.prefix_cacheable = true;
        let capacity_tokens = prompt_tokens
            .len()
            .checked_add(reserved_tokens)
            .ok_or_else(|| RuntimeError::KvCache("session token capacity overflow".into()))?;
        let probe = cache.probe_prefix_recorded(&self.model, prompt_tokens);
        retain_prefix(cache, &probe.cached_blocks)?;
        let capacity_blocks = capacity_tokens.div_ceil(cache.block_size());
        let missing_blocks = capacity_blocks.saturating_sub(probe.cached_blocks.len());
        let mut table = BlockTable::with_block_size(cache.block_size());
        for block in probe.cached_blocks.iter().copied() {
            table.push(block);
        }
        table.set_token_len(probe.cached_tokens);
        self.table = table;
        self.tokens = prompt_tokens.to_vec();
        self.committed_blocks = probe.cached_blocks.len();
        self.last_hash = probe.last_hash;
        Ok(KvPrefillPlan {
            session_id: self.session_id,
            cached_tokens: probe.cached_tokens,
            missing_tokens: probe.missing_tokens,
            write_offset: probe.cached_tokens,
            capacity_blocks,
            needs_eviction: missing_blocks > cache.free_blocks(),
        })
    }

    pub fn allocate_prefill_plan_in_place(
        &mut self,
        cache: &mut KvCache,
        plan: KvPrefillPlan,
    ) -> Result<KvPrefillStep> {
        if plan.session_id != self.session_id
            || plan.cached_tokens != self.table.token_len()
            || plan.missing_tokens != self.tokens.len().saturating_sub(plan.cached_tokens)
        {
            return Err(RuntimeError::KvCache("prefill plan does not match the session".into()));
        }
        let missing_blocks = plan.capacity_blocks.saturating_sub(self.table.blocks().len());
        let allocated = cache.allocate_blocks(missing_blocks)?;
        for block in allocated.blocks().iter().copied() {
            self.table.push(block);
        }
        self.table.set_token_len(self.tokens.len());
        Ok(KvPrefillStep {
            session_id: plan.session_id,
            cached_tokens: plan.cached_tokens,
            missing_tokens: plan.missing_tokens,
            write_offset: plan.write_offset,
        })
    }

    pub fn commit_ready_prefix_blocks(&mut self, cache: &mut KvCache) -> Result<usize> {
        if !self.prefix_cacheable {
            return Ok(0);
        }
        let block_size = cache.block_size();
        let mut committed = 0;
        while self
            .committed_blocks
            .checked_add(1)
            .and_then(|blocks| blocks.checked_mul(block_size))
            .is_some_and(|end| end <= self.tokens.len())
        {
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
}

fn retain_prefix(cache: &mut KvCache, blocks: &[BlockId]) -> Result<()> {
    for block in blocks {
        cache.retain(*block)?;
    }
    Ok(())
}
