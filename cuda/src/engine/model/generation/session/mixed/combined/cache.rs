use std::collections::VecDeque;

use crate::{
    CudaSharedRoutedModelTemplate, Error, Result,
    backend::{
        CudaSharedRoutedPrefillBatch, DEFAULT_PREFILL_CHUNK_TOKENS, RETAINED_CHUNK_SLACK_TOKENS,
        SMALL_PLAN_TOKENS,
    },
};

struct RetainedBatch {
    counts: Vec<usize>,
    capacity: usize,
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
        let batch = template.prepare_padded_prefill_batch(counts, capacity)?;
        self.batches
            .push_back(RetainedBatch { counts: counts.to_vec(), capacity, batch });
        tracing::debug!(
            tokens,
            capacity,
            retained_tokens = self.retained_tokens(),
            retained_shapes = self.batches.len(),
            budget,
            "prepared retained CUDA combined batch"
        );
        Ok(())
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
