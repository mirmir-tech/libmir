use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{
    DecodeDenseSwiGlu, DenseFeedForward, DenseScratch, DenseSwiGluConfig, DenseSwiGluWeights,
    validate,
};
use crate::{CudaBackend, Error, PagedPrefillBatch, PrefillAttentionBf16, Result};

pub struct PrefillDenseSwiGlu {
    attention: PrefillAttentionBf16,
    feed_forward: DenseFeedForward,
    scratch: DenseScratch,
    tokens: usize,
}

impl PrefillDenseSwiGlu {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        tokens: usize,
        weights: DenseSwiGluWeights<'_>,
    ) -> Result<Self> {
        validate(config)?;
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("dense prefill chunk is empty"));
        }
        Ok(Self {
            attention: PrefillAttentionBf16::new(
                backend,
                config.attention,
                tokens,
                Some(weights.attention),
            )?,
            feed_forward: DenseFeedForward::new(backend, config, tokens, weights)?,
            scratch: DenseScratch::new(backend, config, tokens)?,
            tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        state: &mut DecodeDenseSwiGlu,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.attention.execute(
            &mut state.attention,
            input,
            weights.attention,
            write_plan,
            table,
            start_position,
            &mut self.scratch.attention,
        )?;
        self.feed_forward.execute(input, weights, &mut self.scratch, output)
    }

    pub fn execute_batch(
        &mut self,
        state: &mut DecodeDenseSwiGlu,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.attention.execute_batch(
            &mut state.attention,
            input,
            weights.attention,
            batch,
            &mut self.scratch.attention,
        )?;
        self.feed_forward.execute(input, weights, &mut self.scratch, output)
    }

    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }
}
