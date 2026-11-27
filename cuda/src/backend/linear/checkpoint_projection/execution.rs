use mircuda::{DeviceBuffer, bf16};

use super::{CheckpointProjection, CheckpointProjectionWeight};
use crate::{Error, Result};

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
            Self::Ggml { operation, weight } => operation.execute(input, weight, output),
            Self::MxFp8 { operation, weight } => operation.execute(input, weight, output),
            Self::NvFp4 { operation, .. } => operation.execute(input, output),
            Self::NvFp4WeightOnly { operation } => operation.execute(input, output),
            Self::PackedInteger { operation, weight } => operation.execute(input, weight, output),
        }
    }

    /// Executes with another layer's `weight` of the same storage as the one
    /// this projection was planned for, so layers can share one plan. NVFP4
    /// plans keep their prepared weights and refuse.
    pub(in crate::backend) fn execute_with(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &CheckpointProjectionWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        type Weight = CheckpointProjectionWeight;
        self.require_layout(weight)?;
        match (self, weight) {
            (Self::Affine { operation, .. }, Weight::Affine(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::Dense { operation, .. }, Weight::Dense(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::DirectFp8 { operation, .. }, Weight::DirectFp8(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::MxFp4 { operation, .. }, Weight::MxFp4(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::Ggml { operation, .. }, Weight::Ggml(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::MxFp8 { operation, .. }, Weight::MxFp8(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::PackedInteger { operation, .. }, Weight::PackedInteger(weight)) => {
                operation.execute(input, weight, output)
            },
            _ => Err(Error::InvalidExecutionPlan(
                "a late-bound projection weight differs from its plan's storage",
            )),
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
