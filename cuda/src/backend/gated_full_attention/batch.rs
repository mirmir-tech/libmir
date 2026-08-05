use mircuda::{DeviceBuffer, bf16};

use super::{
    AffineGatedFullAttentionConfig, CudaAffineGatedFullAttentionExecution,
    CudaAffineGatedFullAttentionState,
};
use crate::{
    BatchedPagedAttentionBf16, CudaBackend, Error, PagedDecodeBatch, PagedKvCache, Result,
    kernels::BatchedSplitAttentionWorkspace,
};

#[derive(Debug)]
pub(super) struct GatedFullAttentionBatch {
    cache: PagedKvCache,
    attention: BatchedPagedAttentionBf16,
    rows: usize,
}

impl CudaAffineGatedFullAttentionExecution {
    pub(crate) fn prepare_packed(
        &mut self,
        states: &[&mut CudaAffineGatedFullAttentionState],
        paging: &PagedDecodeBatch,
    ) -> Result<()> {
        if states.len() != self.tokens || paging.active() != self.tokens {
            return Err(Error::InvalidDecoderKernel("gated attention packed row mismatch"));
        }
        if self.batch.is_none() {
            self.batch = Some(GatedFullAttentionBatch::new(
                &self.backend,
                states[0],
                self.config,
                paging,
                self.tokens,
                self.batch_workspace.take(),
            )?);
        }
        self.batch
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("gated attention batch was not prepared"))?
            .prepare(states, paging)
    }

    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        paging: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if positions.len() != 3 * self.tokens
            || input.len() != self.tokens * self.config.hidden_size
            || output.len() != input.len()
        {
            return Err(Error::InvalidDecoderKernel("gated attention packed shape mismatch"));
        }
        self.project_and_transform(input, positions)?;
        self.batch
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("gated attention batch was not prepared"))?
            .execute_prepared(
                &self.scratch.rotated_query,
                &self.scratch.rotated_key,
                &self.scratch.value,
                paging,
                &mut self.scratch.attended,
                self.config.attention_scale,
            )?;
        self.gate.execute(
            &self.backend.inner.stream,
            &self.scratch.attended,
            &self.scratch.gate,
            &mut self.scratch.gated,
        )?;
        self.output.execute(&self.scratch.gated, output)
    }

    pub(crate) fn packed_capture_partitions(&self, paging: &PagedDecodeBatch) -> usize {
        self.batch.as_ref().map_or(0, |batch| batch.capture_partitions(paging))
    }
}

impl GatedFullAttentionBatch {
    pub(super) fn new(
        backend: &CudaBackend,
        state: &CudaAffineGatedFullAttentionState,
        config: AffineGatedFullAttentionConfig,
        paging: &PagedDecodeBatch,
        rows: usize,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<Self> {
        Ok(Self {
            cache: state.cache.clone(),
            attention: BatchedPagedAttentionBf16::new_with_workspace(
                backend,
                &state.cache,
                config.query_heads,
                paging.max_blocks(),
                rows,
                workspace,
            )?,
            rows,
        })
    }

    pub(super) fn prepare(
        &self,
        states: &[&mut CudaAffineGatedFullAttentionState],
        paging: &PagedDecodeBatch,
    ) -> Result<()> {
        if states.len() != self.rows || paging.active() != self.rows {
            return Err(Error::InvalidPagedKv("gated attention batch row mismatch"));
        }
        let storage = states[0].cache.storage_spec();
        if states.iter().any(|state| state.cache.storage_spec() != storage) {
            return Err(Error::InvalidPagedKv("gated attention batch cache geometry differs"));
        }
        if paging.cache_config() != storage.cache {
            return Err(Error::InvalidPagedKv("gated attention paging geometry differs"));
        }
        Ok(())
    }

    pub(super) fn execute_prepared(
        &mut self,
        query: &DeviceBuffer<bf16>,
        key: &DeviceBuffer<bf16>,
        value: &DeviceBuffer<bf16>,
        paging: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
        scale: f32,
    ) -> Result<()> {
        self.cache.store_batch(paging, key, value)?;
        self.attention.execute(query, &self.cache, paging, output, None, scale)
    }

    fn capture_partitions(&self, paging: &PagedDecodeBatch) -> usize {
        self.attention.capture_partitions(paging)
    }
}
