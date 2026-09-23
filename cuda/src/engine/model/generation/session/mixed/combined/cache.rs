use std::collections::VecDeque;

use crate::{
    CudaSharedRoutedModelTemplate, Error, Result,
    backend::{
        CudaSharedRoutedPrefillBatch, DEFAULT_PREFILL_CHUNK_TOKENS, RETAINED_CHUNK_SLACK_TOKENS,
        SMALL_PLAN_TOKENS,
    },
};

/// Batches at least this long price the retention headroom per token.
const PRICED_BATCH_TOKENS: usize = 256;

struct RetainedBatch {
    counts: Vec<usize>,
    capacity: usize,
    bytes: u64,
    batch: CudaSharedRoutedPrefillBatch,
}

#[derive(Default)]
pub(in super::super) struct CombinedBatches {
    batches: VecDeque<RetainedBatch>,
}

impl CombinedBatches {
    pub(in super::super) fn clear(&mut self) {
        self.batches.clear();
    }

    pub(super) fn prepare(
        &mut self,
        template: &CudaSharedRoutedModelTemplate,
        counts: &[usize],
    ) -> Result<()> {
        let tokens = counts.iter().sum::<usize>();
        let capacity = if tokens > 512 {
            tokens
                .checked_next_multiple_of(64)
                .ok_or(Error::InvalidExecutionPlan("mixed batch capacity overflow"))?
        } else {
            tokens
        };
        if let Some(index) = self.batches.iter().position(|batch| batch.capacity == capacity) {
            let mut retained = self
                .batches
                .remove(index)
                .ok_or(Error::InvalidExecutionPlan("retained mixed batch disappeared"))?;
            if retained.counts != counts {
                retained.batch.reconfigure(template, counts)?;
                retained.counts = counts.to_vec();
            }
            self.batches.push_back(retained);
            return Ok(());
        }
        // Keep the scalar-plan aggregate limit. Evict before constructing the
        // new shape, so checkpoint tails do not double the scratch peak.
        let budget = DEFAULT_PREFILL_CHUNK_TOKENS
            .max(capacity)
            .saturating_add(RETAINED_CHUNK_SLACK_TOKENS);
        while self.batches.len() >= 8 || self.retained_tokens().saturating_add(capacity) > budget {
            if self.batches.pop_front().is_none() {
                break;
            }
        }
        let used_before = template.pool_used_bytes()?;
        let batch = template.prepare_padded_prefill_batch(counts, capacity)?;
        let bytes = template.pool_used_bytes()?.saturating_sub(used_before);
        self.batches.push_back(RetainedBatch {
            counts: counts.to_vec(),
            capacity,
            bytes,
            batch,
        });
        tracing::debug!(
            tokens,
            capacity,
            bytes,
            pool_used_bytes = template.pool_used_bytes()?,
            retained_tokens = self.retained_tokens(),
            retained_shapes = self.batches.len(),
            budget,
            "prepared retained CUDA combined batch"
        );
        Ok(())
    }

    pub(in super::super) fn follow_cache_blocks(&mut self, block_count: u32) {
        for retained in &mut self.batches {
            retained.batch.follow_cache_blocks(block_count);
        }
    }

    /// Pool bytes shape churn may take beyond the batches held now; before
    /// any priced batch exists, `fallback_per_token` (the plan cost) stands in.
    pub(in super::super) fn headroom_bytes(&self, fallback_per_token: u64) -> u64 {
        // Weighted by capacity: a ten-row, ten-token batch is all per-row
        // overhead and would price a token at a hundred times its cost.
        let (bytes, tokens) = self
            .batches
            .iter()
            .filter(|batch| batch.capacity >= PRICED_BATCH_TOKENS)
            .fold((0_u64, 0_u64), |(bytes, tokens), batch| {
                (
                    bytes.saturating_add(batch.bytes),
                    tokens.saturating_add(u64::try_from(batch.capacity).unwrap_or(u64::MAX)),
                )
            });
        let per_token = bytes.checked_div(tokens).unwrap_or(fallback_per_token);
        let budget =
            u64::try_from(DEFAULT_PREFILL_CHUNK_TOKENS.saturating_add(RETAINED_CHUNK_SLACK_TOKENS))
                .unwrap_or(u64::MAX);
        // Evicted batches stay cached but seldom fit the next shape, so the
        // whole budget can be held twice under traffic.
        budget.saturating_mul(per_token)
    }

    pub(super) fn current(&mut self) -> Result<&mut CudaSharedRoutedPrefillBatch> {
        self.batches
            .back_mut()
            .map(|retained| &mut retained.batch)
            .ok_or(Error::InvalidExecutionPlan("mixed step batch is missing"))
    }

    fn retained_tokens(&self) -> usize {
        self.batches
            .iter()
            .filter(|batch| batch.capacity > SMALL_PLAN_TOKENS)
            .fold(0_usize, |total, batch| total.saturating_add(batch.capacity))
    }
}
