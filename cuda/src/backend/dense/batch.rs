use mircuda::{DeviceBuffer, bf16};

use super::{
    DenseFeedForward, DenseScratch, DenseSwiGluConfig, DenseSwiGluLayerTemplate,
    DenseSwiGluWeights, validate,
};
use crate::{
    BatchedDecodeAttentionBf16, CudaBackend, Error, PagedDecodeBatch, PagedKvCache, Result,
    kernels::BatchedSplitAttentionWorkspace,
};

/// Fixed-shape dense `SwiGLU` layer for independent decode rows.
pub struct BatchedDecodeDenseSwiGlu {
    attention: BatchedDecodeAttentionBf16,
    feed_forward: DenseFeedForward,
    scratch: DenseScratch,
    rows: usize,
}

/// Batched dense layer retaining immutable checkpoint weights.
pub struct BatchedDecodeDenseLayer {
    block: BatchedDecodeDenseSwiGlu,
    template: DenseSwiGluLayerTemplate,
}

impl BatchedDecodeDenseSwiGlu {
    fn new_with_cache(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        rows: usize,
        cache: PagedKvCache,
        weights: DenseSwiGluWeights<'_>,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<Self> {
        validate(config)?;
        if rows == 0 {
            return Err(Error::InvalidDecoderKernel("batched dense decode is empty"));
        }
        Ok(Self {
            attention: BatchedDecodeAttentionBf16::new_with_cache_weights_workspace(
                backend,
                config.attention,
                rows,
                cache,
                Some(weights.attention),
                workspace,
            )?,
            feed_forward: DenseFeedForward::new(backend, config, rows, weights)?,
            scratch: DenseScratch::new(backend, config, rows)?,
            rows,
        })
    }

    fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if batch.active() != self.rows {
            return Err(Error::InvalidDecoderKernel("dense decode batch differs from plan"));
        }
        self.attention
            .execute(input, weights.attention, batch, &mut self.scratch.attention)?;
        self.feed_forward.execute(input, weights, &mut self.scratch, output)
    }
}

impl BatchedDecodeDenseLayer {
    pub(super) fn new(
        backend: &CudaBackend,
        template: DenseSwiGluLayerTemplate,
        rows: usize,
        cache: PagedKvCache,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<Self> {
        let block = BatchedDecodeDenseSwiGlu::new_with_cache(
            backend,
            template.config(),
            rows,
            cache,
            template.weights(),
            workspace,
        )?;
        Ok(Self { block, template })
    }

    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.block.execute(input, self.template.weights(), batch, output)
    }

    /// Split-attention partitions a captured graph of this layer launches.
    pub(in crate::backend) fn capture_partitions(&self, batch: &PagedDecodeBatch) -> usize {
        self.block.attention.capture_partitions(batch)
    }
}
