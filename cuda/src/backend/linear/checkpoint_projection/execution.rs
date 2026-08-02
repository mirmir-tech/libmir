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
}
