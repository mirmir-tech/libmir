use mircuda::{DeviceBuffer, bf16};

use super::{CudaAffineGatedFullAttentionMoeExecution, CudaAffineGatedFullAttentionState};
use crate::{Error, PagedDecodeBatch, Result};

impl CudaAffineGatedFullAttentionMoeExecution {
    /// Follows a K/V resize to `block_count` blocks.
    pub(crate) fn follow_cache_blocks(&mut self, block_count: u32) {
        self.attention.follow_cache_blocks(block_count);
    }

    pub(crate) fn prepare_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        states: &[&mut CudaAffineGatedFullAttentionState],
        paging: &PagedDecodeBatch,
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        self.attention.prepare_packed(states, paging)
    }

    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        paging: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            input,
            bf16_tensor(&self.input_norm_weight)?,
            &mut scratch.normalized,
        )?;
        self.attention.execute_prepared_packed(
            &scratch.normalized,
            positions,
            paging,
            &mut scratch.attention,
        )?;
        self.residual.add(stream, input, &scratch.attention, &mut scratch.residual)?;
        self.post_attention_norm.execute(
            stream,
            &scratch.residual,
            bf16_tensor(&self.post_attention_norm_weight)?,
            &mut scratch.normalized,
        )?;
        self.moe.execute(&scratch.normalized, &mut scratch.moe)?;
        let result = self.residual.add(stream, &scratch.residual, &scratch.moe, output);
        drop(guard);
        result
    }

    pub(crate) fn packed_capture_partitions(&self, paging: &PagedDecodeBatch) -> usize {
        self.attention.packed_capture_partitions(paging)
    }
}

fn bf16_tensor(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
