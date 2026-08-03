use mircuda::{DeviceBuffer, FmhaBf16Plan, FmhaBf16Spec, bf16};

use super::{CudaAffineGatedFullAttentionExecution, CudaAffineGatedFullAttentionState};
use crate::{CudaBackend, Error, PagedPrefillBatch, Result};

#[derive(Debug)]
pub(super) struct GatedFullAttentionPrefill {
    attention: FmhaBf16Plan,
    workspace: DeviceBuffer<bf16>,
}

impl GatedFullAttentionPrefill {
    fn new(
        backend: &CudaBackend,
        execution: &CudaAffineGatedFullAttentionExecution,
    ) -> Result<Self> {
        let config = execution.config;
        let spec = FmhaBf16Spec::new(
            config.query_heads,
            config.key_value_heads,
            config.head_dim,
            config.head_dim,
        )?;
        let elements = execution
            .tokens
            .checked_mul(config.hidden_size)
            .ok_or(Error::InvalidPagedKv("gated prefill workspace overflow"))?;
        Ok(Self {
            attention: FmhaBf16Plan::new(&backend.inner.context, &backend.inner.stream, spec)?,
            workspace: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
        })
    }
}

impl CudaAffineGatedFullAttentionExecution {
    pub(crate) fn execute_packed_prefill(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        states: &mut [&mut CudaAffineGatedFullAttentionState],
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if states.len() != batch.active()
            || batch.tokens() != self.tokens
            || positions.len() != 3 * self.tokens
        {
            return Err(Error::InvalidPagedKv("gated packed prefill shape mismatch"));
        }
        let storage = states
            .first()
            .ok_or(Error::InvalidPagedKv("gated packed prefill has no state"))?
            .storage_spec();
        if states.iter().any(|state| state.storage_spec() != storage) {
            return Err(Error::InvalidPagedKv("gated packed prefill cache geometry differs"));
        }
        self.project_and_transform(input, positions)?;
        let state = &mut states[0];
        state
            .cache
            .store_prefill_batch(batch, &self.scratch.rotated_key, &self.scratch.value)?;
        if self.prefill.is_none() {
            self.prefill = Some(GatedFullAttentionPrefill::new(&self.backend, self)?);
        }
        let prefill = self
            .prefill
            .as_mut()
            .ok_or(Error::InvalidPagedKv("missing gated packed prefill plan"))?;
        prefill.attention.execute_paged_varlen(
            &self.backend.inner.stream,
            &self.scratch.rotated_query,
            state.cache.key_pages(),
            state.cache.value_pages(),
            &mut self.scratch.attended,
            batch.query_starts(),
            batch.token_counts(),
            batch.context_starts(),
            batch.tables(),
            &mut prefill.workspace,
            batch.active(),
            batch.tokens(),
            batch.max_query_tokens(),
            batch.max_context_tokens(),
            batch.max_blocks(),
            batch.cache_config().block_size,
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
}
