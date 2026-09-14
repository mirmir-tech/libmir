use std::collections::VecDeque;

use crate::{
    CudaSharedRoutedModelTemplate, Error, Result,
    backend::{CudaSharedRoutedPrefillBatch, DEFAULT_PREFILL_CHUNK_TOKENS},
};

struct RetainedBatch {
    counts: Vec<usize>,
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
        if let Some(index) = self
            .batches
            .iter()
            .position(|batch| batch.counts.iter().sum::<usize>() == tokens)
        {
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
        let budget = DEFAULT_PREFILL_CHUNK_TOKENS.max(tokens).saturating_add(1);
        while self.batches.len() >= 8 || self.retained_tokens().saturating_add(tokens) > budget {
            if self.batches.pop_front().is_none() {
                break;
            }
        }
        let batch = template.prepare_ragged_prefill_batch(counts)?;
        self.batches.push_back(RetainedBatch { counts: counts.to_vec(), batch });
        tracing::debug!(
            tokens,
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
            .flat_map(|batch| &batch.counts)
            .fold(0_usize, |total, count| total.saturating_add(*count))
    }
}
