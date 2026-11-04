use mircuda::{DeviceBuffer, bf16};

use super::BatchedDecodeMoeBlockBf16;
use crate::{PagedDecodeBatch, Result, backend::block::graph::weights::CapturedBlockWeights};

/// One batched layer retaining immutable checkpoint weights.
pub struct BatchedDecodeMoeLayer {
    block: BatchedDecodeMoeBlockBf16,
    weights: CapturedBlockWeights,
}

impl BatchedDecodeMoeLayer {
    pub(in crate::backend::block) const fn new(
        block: BatchedDecodeMoeBlockBf16,
        weights: CapturedBlockWeights,
    ) -> Self {
        Self { block, weights }
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.block.execute(input, self.weights.borrow(), batch, output)
    }

    /// Split-attention partitions a captured graph of this layer launches.
    pub(in crate::backend) fn capture_partitions(&self, batch: &PagedDecodeBatch) -> usize {
        self.block.capture_partitions(batch)
    }
}
