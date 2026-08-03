use mircuda::{DeviceBuffer, bf16};
use runtime::kv::BlockTable;

use super::{
    AffineGatedFullAttentionConfig, CudaAffineGatedFullAttentionExecution,
    CudaAffineGatedFullAttentionState,
};
use crate::{
    BatchedPagedAttentionBf16, CudaBackend, Error, PagedDecodeBatch, PagedKvCache, Result,
};

#[derive(Debug)]
pub(super) struct GatedFullAttentionBatch {
    cache: PagedKvCache,
    paging: PagedDecodeBatch,
    attention: BatchedPagedAttentionBf16,
    rows: usize,
}

impl CudaAffineGatedFullAttentionExecution {
    pub(crate) fn prepare_packed(
        &mut self,
        states: &[&mut CudaAffineGatedFullAttentionState],
        tables: &[&BlockTable],
        max_blocks: usize,
    ) -> Result<()> {
        if states.len() != self.tokens || tables.len() != self.tokens {
            return Err(Error::InvalidDecoderKernel("gated attention packed row mismatch"));
        }
        if self.batch.is_none() {
            self.batch = Some(GatedFullAttentionBatch::new(
                &self.backend, states[0], self.config, max_blocks, self.tokens,
            )?);
        }
        self.batch
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("gated attention batch was not prepared"))?
            .prepare(states, tables)
    }

    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
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

    pub(crate) fn packed_capture_partitions(&self) -> usize {
        self.batch.as_ref().map_or(0, GatedFullAttentionBatch::capture_partitions)
    }
}

impl GatedFullAttentionBatch {
    pub(super) fn new(
        backend: &CudaBackend,
        state: &CudaAffineGatedFullAttentionState,
        config: AffineGatedFullAttentionConfig,
        max_blocks: usize,
        rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            cache: state.cache.clone(),
            paging: backend.prepare_paged_decode_batch(
                state.cache.storage_spec(),
                max_blocks,
                rows,
            )?,
            attention: backend.prepare_batched_paged_attention_bf16(
                &state.cache,
                config.query_heads,
                max_blocks,
                rows,
            )?,
            rows,
        })
    }

    pub(super) fn prepare(
        &mut self,
        states: &[&mut CudaAffineGatedFullAttentionState],
        tables: &[&BlockTable],
    ) -> Result<()> {
        if states.len() != self.rows || tables.len() != self.rows {
            return Err(Error::InvalidPagedKv("gated attention batch row mismatch"));
        }
        let storage = states[0].cache.storage_spec();
        if states.iter().any(|state| state.cache.storage_spec() != storage) {
            return Err(Error::InvalidPagedKv("gated attention batch cache geometry differs"));
        }
        self.paging.prepare(tables)
    }

    pub(super) fn execute_prepared(
        &mut self,
        query: &DeviceBuffer<bf16>,
        key: &DeviceBuffer<bf16>,
        value: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        scale: f32,
    ) -> Result<()> {
        self.cache.store_batch(&self.paging, key, value)?;
        self.attention.execute(query, &self.cache, &self.paging, output, None, scale)
    }

    fn capture_partitions(&self) -> usize {
        self.attention.capture_partitions(&self.paging)
    }
}
