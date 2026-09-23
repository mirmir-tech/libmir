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
    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    pub(crate) fn execute_packed_prefill(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        states: &mut [&mut CudaAffineGatedFullAttentionState],
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if states.len() != batch.active()
            || batch.tokens() > self.tokens
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
        let shared = self.scratch.clone_handle();
        let mut guard = shared.lock()?;
        let scratch = &mut *guard;
        self.project_and_transform(scratch, input, positions)?;
        let state = &mut *states[0];
        state.cache.store_prefill_batch(batch, &scratch.rotated_key, &scratch.value)?;
        if self.prefill.is_none() {
            self.prefill = Some(GatedFullAttentionPrefill::new(&self.backend, self)?);
        }
        let prefill = self
            .prefill
            .as_mut()
            .ok_or(Error::InvalidPagedKv("missing gated packed prefill plan"))?;
        let active_query = super::checked(batch.tokens(), self.config.query_width()?)?;
        prefill.attention.execute_paged_varlen(
            &self.backend.inner.stream,
            &scratch.rotated_query.slice(0..active_query)?,
            state.cache.key_pages(),
            state.cache.value_pages(),
            &mut scratch.attended.slice(0..active_query)?,
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
            &scratch.attended,
            &scratch.gate,
            &mut scratch.gated,
        )?;
        self.output.execute(&scratch.gated, output)
    }
}
