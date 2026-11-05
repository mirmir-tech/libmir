use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{DenseFeedForward, DenseScratch, DenseSwiGluConfig, DenseSwiGluWeights};
use crate::{CudaBackend, DecodeAttentionBf16, Error, Result};

pub struct DecodeDenseSwiGlu {
    pub(in crate::backend) attention: DecodeAttentionBf16,
    pub(super) feed_forward: DenseFeedForward,
    pub(super) scratch: DenseScratch,
}

impl DecodeDenseSwiGlu {
    pub(in crate::backend) fn new_with_cache(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        cache: crate::PagedKvCache,
        weights: DenseSwiGluWeights<'_>,
    ) -> Result<Self> {
        validate(config)?;
        Ok(Self {
            attention: DecodeAttentionBf16::new_with_cache_and_weights(
                backend,
                config.attention,
                cache,
                Some(weights.attention),
            )?,
            feed_forward: DenseFeedForward::new(backend, config, 1, weights)?,
            scratch: DenseScratch::new(backend, config, 1)?,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.attention.execute(
            input,
            weights.attention,
            write_plan,
            table,
            &mut self.scratch.attention,
        )?;
        self.feed_forward.execute(input, weights, &mut self.scratch, output)
    }
}

pub(super) fn validate(config: DenseSwiGluConfig) -> Result<()> {
    if config.intermediate_size == 0 {
        Err(Error::InvalidDecoderKernel("dense SwiGLU intermediate size is zero"))
    } else {
        Ok(())
    }
}
