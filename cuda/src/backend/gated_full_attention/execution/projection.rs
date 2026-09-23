use mircuda::{DeviceBuffer, bf16};

use super::{super::scratch::GatedAttentionScratch, CudaAffineGatedFullAttentionExecution};
use crate::{CudaTensor, Error, Result};

impl CudaAffineGatedFullAttentionExecution {
    pub(in crate::backend::gated_full_attention) fn project_and_transform(
        &mut self,
        scratch: &mut GatedAttentionScratch,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        match (&mut self.packed_qkv, &self.packed_split, &mut scratch.packed_qkv) {
            (Some(projection), Some(split), Some(packed)) => {
                projection.execute(input, packed)?;
                split.execute3(
                    stream,
                    packed,
                    &mut scratch.query_projected,
                    &mut scratch.key,
                    &mut scratch.value,
                )?;
            },
            (None, None, None) => {
                self.query
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("attention query projection is missing"))?
                    .execute(input, &mut scratch.query_projected)?;
                self.key
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("attention key projection is missing"))?
                    .execute(input, &mut scratch.key)?;
                self.value
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("attention value projection is missing"))?
                    .execute(input, &mut scratch.value)?;
            },
            _ => {
                return Err(Error::InvalidExecutionPlan(
                    "packed attention projection contract is incomplete",
                ));
            },
        }
        self.transform.execute(
            stream,
            &scratch.query_projected,
            &scratch.key,
            bf16_tensor(&self.weights.query_norm)?,
            bf16_tensor(&self.weights.key_norm)?,
            positions,
            &mut scratch.rotated_query,
            &mut scratch.rotated_key,
            &mut scratch.gate,
        )
    }
}

fn bf16_tensor(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
