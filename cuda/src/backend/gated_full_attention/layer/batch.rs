use mircuda::{DeviceBuffer, bf16};

use super::{CudaAffineGatedFullAttentionMoeExecution, CudaAffineGatedFullAttentionState};
use crate::{Error, PagedDecodeBatch, Result};

impl CudaAffineGatedFullAttentionMoeExecution {
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
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            input,
            bf16_tensor(&self.input_norm_weight)?,
            &mut self.scratch.normalized,
        )?;
        self.attention.execute_prepared_packed(
            &self.scratch.normalized,
            positions,
            paging,
            &mut self.scratch.attention,
        )?;
        self.residual
            .add(stream, input, &self.scratch.attention, &mut self.scratch.residual)?;
        self.post_attention_norm.execute(
            stream,
            &self.scratch.residual,
            bf16_tensor(&self.post_attention_norm_weight)?,
            &mut self.scratch.normalized,
        )?;
        self.moe.execute(&self.scratch.normalized, &mut self.scratch.moe)?;
        self.residual.add(stream, &self.scratch.residual, &self.scratch.moe, output)
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
