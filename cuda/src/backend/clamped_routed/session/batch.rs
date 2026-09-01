use std::collections::HashSet;

use runtime::{
    backend::SamplingLogits,
    kv::{BlockTable, KvStorageSpec},
};
use uuid::Uuid;

use super::CudaClampedRoutedModelSession;
use crate::{Error, Result};

const MAX_PACKED_BATCH_SHAPES: usize = 8;

impl CudaClampedRoutedModelSession {
    pub(crate) fn prefill_packed_chunk(
        &mut self,
        sessions: &[Uuid],
        tokens: &[u32],
        tables: &[&BlockTable],
        starts: &[usize],
        counts: &[usize],
        write_starts: &[usize],
    ) -> Result<()> {
        self.last_packed_decode = None;
        self.validate_packed(sessions, tokens, tables, starts, counts)?;
        if write_starts.len() != sessions.len() {
            return Err(Error::InvalidPagedKv(
                "packed cached write offsets differ from session rows",
            ));
        }
        for token in tokens {
            self.embedding.validate_token(*token)?;
        }
        let ring_slots = self.rings.acquire_many(sessions)?;
        let key = (sessions.len(), tokens.len());
        let mut batch = self
            .packed_batches
            .remove(&key)
            .map_or_else(|| self.new_packed_batch(key.0, key.1), Ok)?;
        batch.prepare(tables, starts, counts)?;
        batch.skip_cached_slot_writes(write_starts)?;
        if let Some(ring_window) = self.template.max_sliding_window() {
            let ring_blocks = self
                .template
                .ring_blocks()
                .ok_or(Error::InvalidPagedKv("missing windowed KV ring geometry"))?;
            batch.prepare_ring(tables, starts, counts, &ring_slots, ring_blocks, ring_window)?;
        }
        if !self.plans.contains_key(&tokens.len()) {
            self.plans.clear();
            let plan = super::super::plan::ClampedRoutedExecutionPlan::new(
                &self.template,
                tokens.len(),
                crate::ExecutionPhase::Prefill,
            )?;
            // Packed cohort shapes vary with cache hits. Retaining every exact
            // shape pins proportional activation and attention workspaces.
            self.plans.insert(tokens.len(), plan);
        }
        let plan = self
            .plans
            .get_mut(&tokens.len())
            .ok_or(Error::InvalidDecoderKernel("missing packed clamped-routed execution plan"))?;
        plan.upload_packed(&self.template, tokens)?;
        let result = plan.execute_batch(&self.template, &mut self.state, &self.embedding, &batch);
        if self.packed_batches.len() < MAX_PACKED_BATCH_SHAPES {
            self.packed_batches.insert(key, batch);
        }
        result?;
        #[cfg(feature = "diagnostics")]
        plan.publish_fingerprints()?;
        for ((session, start), count) in sessions.iter().zip(starts).zip(counts) {
            self.positions.insert(*session, start + count);
        }
        Ok(())
    }

    pub(crate) fn finish_packed_prefill_row(
        &mut self,
        row: usize,
        total: usize,
        sampling: SamplingLogits,
    ) -> Result<()> {
        if row >= total {
            return Err(Error::InvalidDecoderKernel(
                "packed clamped-routed output row is out of bounds",
            ));
        }
        let hidden = if self.last_packed_decode == Some(total) {
            self.decode_batches
                .get(&total)
                .ok_or(Error::InvalidDecoderKernel("missing clamped-routed decode batch"))?
                .hidden()?
        } else {
            self.plans
                .get(&total)
                .ok_or(Error::InvalidDecoderKernel("missing packed clamped-routed output plan"))?
                .hidden()
        };
        self.select.execute(
            &self.template.backend.inner.stream,
            hidden,
            &mut self.last_hidden,
            row,
            total,
        )?;
        self.final_norm.execute(
            &self.last_hidden,
            &self.template.final_norm,
            &mut self.normalized,
        )?;
        self.output.execute(&self.normalized, &mut self.logits, sampling)
    }

    pub(super) fn validate_packed(
        &self,
        sessions: &[Uuid],
        tokens: &[u32],
        tables: &[&BlockTable],
        starts: &[usize],
        counts: &[usize],
    ) -> Result<()> {
        let rows = sessions.len();
        let unique = sessions.iter().copied().collect::<HashSet<_>>();
        if rows == 0
            || tokens.is_empty()
            || tables.len() != rows
            || starts.len() != rows
            || counts.len() != rows
            || counts.iter().sum::<usize>() != tokens.len()
            || unique.len() != rows
        {
            return Err(Error::InvalidPagedKv("invalid packed clamped-routed prefill geometry"));
        }
        for (((session, table), start), count) in
            sessions.iter().zip(tables).zip(starts).zip(counts)
        {
            let end = start
                .checked_add(*count)
                .ok_or(Error::InvalidPagedKv("packed prefill position overflow"))?;
            let expected = self.positions.get(session).copied().unwrap_or_default();
            if *count == 0
                || *start != expected
                || table.token_len() != end
                || table.block_size() != Some(self.template.cache.block_size)
            {
                return Err(Error::InvalidPagedKv(
                    "packed clamped-routed prompt differs from session state",
                ));
            }
        }
        Ok(())
    }

    fn new_packed_batch(&self, rows: usize, tokens: usize) -> Result<crate::PagedPrefillBatch> {
        let storage: KvStorageSpec = self.template.config.storage(self.template.cache);
        self.template.backend.prepare_paged_prefill_batch(
            storage,
            self.template.max_sequence_blocks,
            rows,
            tokens,
        )
    }
}
