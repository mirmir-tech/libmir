use super::{CudaSharedRoutedDecodeBatch, CudaSharedRoutedPrefillBatch};
use crate::{CudaSharedRoutedModelTemplate, Error, Result};

impl CudaSharedRoutedModelTemplate {
    pub(crate) fn prepare_decode_batch(&self, rows: usize) -> Result<CudaSharedRoutedDecodeBatch> {
        CudaSharedRoutedDecodeBatch::new(self, rows)
    }

    pub(crate) fn prepare_prefill_batch(
        &self,
        rows: usize,
        row_tokens: usize,
    ) -> Result<CudaSharedRoutedPrefillBatch> {
        CudaSharedRoutedPrefillBatch::new(self, rows, row_tokens)
    }

    pub(crate) fn supports_combined_generation(&self) -> bool {
        self.decoder.num_experts.is_none()
            && self.layers.iter().enumerate().all(|(index, layer)| {
                !matches!(layer, super::super::SharedRoutedLayerTemplate::Full(_))
                    || matches!(self.decoder.layer_head_dim(index), 64 | 128 | 256)
            })
    }

    pub(crate) fn prepare_ragged_prefill_batch(
        &self,
        counts: &[usize],
    ) -> Result<CudaSharedRoutedPrefillBatch> {
        // The mixed batch replaces scalar prefill workspace, rather than keeping
        // another model-sized plan alive alongside it.
        self.plans
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("mixed plan cache lock poisoned"))?
            .clear();
        CudaSharedRoutedPrefillBatch::new_ragged(self, counts)
    }
}
