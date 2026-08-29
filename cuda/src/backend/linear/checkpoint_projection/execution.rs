use mircuda::{DeviceBuffer, bf16};

use super::CheckpointProjection;
use crate::Result;

impl CheckpointProjection {
    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Affine { operation, weight } => operation.execute(input, weight, output),
            Self::Dense { operation, weight } => operation.execute(input, weight, output),
            Self::DirectFp8 { operation, weight } => operation.execute(input, weight, output),
            Self::MxFp4 { operation, weight } => operation.execute(input, weight, output),
            Self::MxFp8 { operation, weight } => operation.execute(input, weight, output),
            Self::NvFp4 { operation, .. } => operation.execute(input, output),
            Self::NvFp4WeightOnly { operation } => operation.execute(input, output),
            Self::PackedInteger { operation, weight } => operation.execute(input, weight, output),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_norm_gate(
        &mut self,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        norm: &DeviceBuffer<bf16>,
        epsilon: f32,
        weight_shift: f32,
        value_heads: usize,
        gate_stride: usize,
        gate_offset: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<bool> {
        let Self::DirectFp8 { operation, weight } = self else {
            return Ok(false);
        };
        operation.execute_norm_gate(
            input, gate, norm, weight, epsilon, weight_shift, value_heads, gate_stride,
            gate_offset, output,
        )
    }
}
