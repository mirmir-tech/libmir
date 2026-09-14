use mircuda::{DeviceBuffer, bf16};

use super::{CudaAffineGatedDeltaExecution, execution::bf16};
use crate::{Error, Result};

impl CudaAffineGatedDeltaExecution {
    pub(super) fn finish_projected(
        &mut self,
        packed: bool,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        let (gate, gate_stride, gate_offset) = if packed {
            (
                self.scratch
                    .packed_qkv_gate
                    .as_ref()
                    .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?,
                self.config.mixed_width()? + self.config.value_width()?,
                self.config.mixed_width()?,
            )
        } else {
            (&self.scratch.gate, self.config.value_width()?, 0)
        };
        if self.config.value_dim == 128
            && self.output.execute_norm_gate(
                &self.scratch.recurrent,
                gate,
                bf16(&self.weights.norm)?,
                self.config.rms_norm_epsilon,
                self.config.norm_weight_shift,
                self.config.value_heads,
                gate_stride,
                gate_offset,
                output,
            )?
        {
            return Ok(());
        }
        if packed {
            self.transforms.norm_gate_strided(
                stream,
                &self.scratch.recurrent,
                self.scratch
                    .packed_qkv_gate
                    .as_ref()
                    .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?,
                bf16(&self.weights.norm)?,
                &mut self.scratch.gated,
                self.config.mixed_width()? + self.config.value_width()?,
                self.config.mixed_width()?,
            )?;
        } else {
            self.transforms.norm_gate(
                stream,
                &self.scratch.recurrent,
                &self.scratch.gate,
                bf16(&self.weights.norm)?,
                &mut self.scratch.gated,
            )?;
        }
        self.output.execute(&self.scratch.gated, output)
    }
}
