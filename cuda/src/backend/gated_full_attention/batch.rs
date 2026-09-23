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
    /// The sessions' pages, rebound after a K/V resize replaces them.
    cache: Option<PagedKvCache>,
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

    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
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
        let shared = self.scratch.clone_handle();
        let mut guard = shared.lock()?;
        let scratch = &mut *guard;
        self.project_and_transform(scratch, input, positions)?;
        self.batch
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("gated attention batch was not prepared"))?
            .execute_prepared(
                &scratch.rotated_query,
                &scratch.rotated_key,
                &scratch.value,
                paging,
                &mut scratch.attended,
                self.config.attention_scale,
            )?;
        self.gate.execute(
            &self.backend.inner.stream,
            &scratch.attended,
            &scratch.gate,
            &mut scratch.gated,
        )?;
        self.output.execute(&scratch.gated, output)
    }

    pub(crate) fn packed_capture_partitions(&self, paging: &PagedDecodeBatch) -> usize {
        self.batch.as_ref().map_or(0, |batch| batch.capture_partitions(paging))
    }

    /// Follows a K/V resize to `block_count` blocks; the next prepare binds
    /// the sessions' new pages.
    pub(crate) fn follow_cache_blocks(&mut self, block_count: u32) {
        if let Some(batch) = &mut self.batch {
            batch.cache = None;
            batch.attention.follow_cache_blocks(block_count);
        }
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
            cache: Some(state.cache.clone()),
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
        &mut self,
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
        if self.cache.as_ref().is_none_or(|cache| cache.storage_spec() != storage) {
            self.cache = Some(states[0].cache.clone());
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
        let cache = self
            .cache
            .as_mut()
            .ok_or(Error::InvalidPagedKv("gated attention batch has no pages bound"))?;
        cache.store_batch(paging, key, value)?;
        self.attention.execute(query, cache, paging, output, None, scale)
    }

    fn capture_partitions(&self, paging: &PagedDecodeBatch) -> usize {
        self.attention.capture_partitions(paging)
    }
}
