//! Packed decode: one token for each of several sequences over the layer's
//! shared pages, captured with the rest of the decode graph.

use mircuda::{DeviceBuffer, bf16};

use super::CudaAttentionHybridExecution;
use crate::{
    BatchedDecodeAttentionBf16, DecodeAttentionBf16, Error, PagedDecodeBatch, Result,
    kernels::BatchedSplitAttentionWorkspace,
};

impl CudaAttentionHybridExecution {
    /// Shares a split-attention workspace among the model's attention layers.
    pub(crate) fn set_workspace(&mut self, workspace: Option<BatchedSplitAttentionWorkspace>) {
        self.workspace = workspace;
    }

    /// Host-side preparation of a packed decode step: binds the sessions'
    /// pages, building the batched attention on first use.
    pub(crate) fn prepare_packed(
        &mut self,
        states: &[&mut DecodeAttentionBf16],
        paging: &PagedDecodeBatch,
    ) -> Result<()> {
        let [first, ..] = states else {
            return Err(Error::InvalidDecoderKernel("attention packed step has no rows"));
        };
        if states.len() != self.tokens || paging.active() != self.tokens {
            return Err(Error::InvalidDecoderKernel("attention packed rows differ from the plan"));
        }
        if let Some(batch) = &mut self.batch {
            return batch.bind_cache(&first.cache);
        }
        self.batch = Some(BatchedDecodeAttentionBf16::new_with_cache_weights_workspace(
            &self.backend,
            self.config,
            self.tokens,
            first.cache.clone(),
            Some(self.weights.view()),
            self.workspace.take(),
        )?);
        Ok(())
    }

    /// Device work of a packed decode step; no allocation or host sync.
    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        paging: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let batch = self
            .batch
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("attention packed step was not prepared"))?;
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        batch.execute(input, self.weights.view(), paging, &mut scratch.attention)?;
        self.tail.execute(input, scratch, output)
    }

    pub(crate) fn packed_capture_partitions(&self, paging: &PagedDecodeBatch) -> usize {
        self.batch.as_ref().map_or(0, |batch| batch.capture_partitions(paging))
    }
}
