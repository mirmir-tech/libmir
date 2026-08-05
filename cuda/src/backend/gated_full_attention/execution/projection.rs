use mircuda::{DeviceBuffer, bf16};

use super::CudaAffineGatedFullAttentionExecution;
use crate::{CudaTensor, Error, Result};

impl CudaAffineGatedFullAttentionExecution {
    pub(in crate::backend::gated_full_attention) fn project_and_transform(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        match (&mut self.packed_qkv, &self.packed_split, &mut self.scratch.packed_qkv) {
            (Some(projection), Some(split), Some(packed)) => {
                projection.execute(input, packed)?;
                split.execute3(
                    stream,
                    packed,
                    &mut self.scratch.query_projected,
                    &mut self.scratch.key,
                    &mut self.scratch.value,
                )?;
            },
            (None, None, None) => {
                self.query
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("attention query projection is missing"))?
                    .execute(input, &mut self.scratch.query_projected)?;
                self.key
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("attention key projection is missing"))?
                    .execute(input, &mut self.scratch.key)?;
                self.value
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("attention value projection is missing"))?
                    .execute(input, &mut self.scratch.value)?;
            },
            _ => {
                return Err(Error::InvalidExecutionPlan(
                    "packed attention projection contract is incomplete",
                ));
            },
        }
        self.split.execute(
            stream,
            &self.scratch.query_projected,
            &mut self.scratch.query,
            &mut self.scratch.gate,
        )?;
        self.query_norm.execute(
            stream,
            &self.scratch.query,
            bf16_tensor(&self.weights.query_norm)?,
            &mut self.scratch.normalized_query,
        )?;
        self.key_norm.execute(
            stream,
            &self.scratch.key,
            bf16_tensor(&self.weights.key_norm)?,
            &mut self.scratch.normalized_key,
        )?;
        self.query_rope.execute(
            stream,
            &self.scratch.normalized_query,
            positions,
            &mut self.scratch.rotated_query,
        )?;
        self.key_rope.execute(
            stream,
            &self.scratch.normalized_key,
            positions,
            &mut self.scratch.rotated_key,
        )
    }
}

fn bf16_tensor(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
