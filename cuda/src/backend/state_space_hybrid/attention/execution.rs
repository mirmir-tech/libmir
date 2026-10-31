use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{AttentionWeights, CudaAttentionHybridLayer};
use crate::{
    BatchedDecodeAttentionBf16, CudaBackend, DecodeAttentionBf16, DecodeAttentionConfig, Error,
    ExecutionPhase, PagedPrefillBatch, PrefillAttentionBf16, Result,
    backend::{layer_scratch::SharedLayerScratch, state_space_hybrid::tail::TailExecution},
    kernels::BatchedSplitAttentionWorkspace,
};

/// Kernels and scratch of one attention layer for a fixed number of rows.
#[derive(Debug)]
pub struct CudaAttentionHybridExecution {
    pub(super) backend: CudaBackend,
    pub(super) config: DecodeAttentionConfig,
    pub(super) tokens: usize,
    pub(super) weights: AttentionWeights,
    prefill: Option<PrefillAttentionBf16>,
    pub(super) batch: Option<BatchedDecodeAttentionBf16>,
    pub(super) workspace: Option<BatchedSplitAttentionWorkspace>,
    pub(super) tail: TailExecution,
    pub(super) scratch: SharedLayerScratch,
}

impl CudaAttentionHybridExecution {
    pub(super) fn new(
        layer: &CudaAttentionHybridLayer,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let backend = &layer.backend;
        let prefill = (tokens > 1)
            .then(|| {
                PrefillAttentionBf16::new(backend, layer.config, tokens, Some(layer.weights.view()))
            })
            .transpose()?;
        Ok(Self {
            backend: backend.clone(),
            config: layer.config,
            tokens,
            weights: layer.weights.clone(),
            prefill,
            batch: None,
            workspace: None,
            tail: TailExecution::new(
                backend,
                layer.norm,
                layer.feed_forward.as_ref(),
                tokens,
                phase,
            )?,
            scratch: backend.inner.layer_scratch.acquire(
                backend,
                tokens,
                layer.norm.hidden_size,
            )?,
        })
    }

    /// Advances one sequence by this execution's rows, writing their K/V at
    /// `start_position` through `write`.
    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &mut DecodeAttentionBf16,
        write: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        let weights = self.weights.view();
        match &mut self.prefill {
            Some(prefill) => {
                prefill.follow_cache(state.config.cache)?;
                prefill.execute(
                    state,
                    input,
                    weights,
                    write,
                    table,
                    start_position,
                    &mut scratch.attention,
                )?;
            },
            None => state.execute(input, weights, write, table, &mut scratch.attention)?,
        }
        self.tail.execute(input, scratch, output)
    }

    /// Prefills several sequences whose rows are concatenated in `input`.
    /// The sequences share the layer's pages; `state` provides them. A
    /// one-row batch (one sequence with one uncached prompt token) still takes
    /// the paged prefill path, so its plan is built on first use.
    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    pub(crate) fn execute_packed_prefill(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &mut DecodeAttentionBf16,
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.prefill.is_none() {
            self.prefill = Some(PrefillAttentionBf16::new(
                &self.backend,
                self.config,
                self.tokens,
                Some(self.weights.view()),
            )?);
        }
        let prefill = self
            .prefill
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("packed prefill plan is missing"))?;
        prefill.follow_cache(state.config.cache)?;
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        prefill.execute_batch(state, input, self.weights.view(), batch, &mut scratch.attention)?;
        self.tail.execute(input, scratch, output)
    }
}
