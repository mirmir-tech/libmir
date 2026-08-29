use mircuda::{DeviceBuffer, bf16};

use super::{DirectFp8Bf16Linear, DirectFp8CheckpointWeight};
use crate::Result;

impl DirectFp8Bf16Linear {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_norm_gate(
        &self,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        norm: &DeviceBuffer<bf16>,
        weight: &DirectFp8CheckpointWeight,
        epsilon: f32,
        weight_shift: f32,
        value_heads: usize,
        gate_stride: usize,
        gate_offset: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<bool> {
        self.validate_weight(weight)?;
        self.operation.execute_norm_gate(
            &self.stream, input, gate, norm, weight, epsilon, weight_shift, value_heads,
            gate_stride, gate_offset, output,
        )
    }
}
