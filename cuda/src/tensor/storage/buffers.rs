//! The device buffers behind a tensor, one per storage type.

use mircuda::{DeviceBuffer, bf16, f16};

use super::CudaTensorDType;
use crate::kernels::GgmlBlocks;

#[derive(Clone, Debug)]
pub(in crate::tensor) enum TensorStorage {
    F16(DeviceBuffer<f16>),
    Bf16(DeviceBuffer<bf16>),
    F32(DeviceBuffer<f32>),
    F8E4M3(DeviceBuffer<u8>),
    F8E5M2(DeviceBuffer<u8>),
    U32(DeviceBuffer<u32>),
    I32(DeviceBuffer<i32>),
    U8(DeviceBuffer<u8>),
    /// GGML blocks in their padded allocation; their dtype is U8.
    GgmlBlocks(GgmlBlocks),
    I8(DeviceBuffer<i8>),
}

impl TensorStorage {
    pub(super) const fn dtype(&self) -> CudaTensorDType {
        match self {
            Self::F16(_) => CudaTensorDType::F16,
            Self::Bf16(_) => CudaTensorDType::Bf16,
            Self::F32(_) => CudaTensorDType::F32,
            Self::F8E4M3(_) => CudaTensorDType::F8E4M3,
            Self::F8E5M2(_) => CudaTensorDType::F8E5M2,
            Self::U32(_) => CudaTensorDType::U32,
            Self::I32(_) => CudaTensorDType::I32,
            Self::U8(_) | Self::GgmlBlocks(_) => CudaTensorDType::U8,
            Self::I8(_) => CudaTensorDType::I8,
        }
    }

    pub(super) fn bytes(&self) -> usize {
        match self {
            Self::F16(buffer) => buffer.bytes(),
            Self::Bf16(buffer) => buffer.bytes(),
            Self::F32(buffer) => buffer.bytes(),
            Self::F8E4M3(buffer) | Self::F8E5M2(buffer) | Self::U8(buffer) => buffer.bytes(),
            Self::GgmlBlocks(blocks) => blocks.view().bytes(),
            Self::U32(buffer) => buffer.bytes(),
            Self::I32(buffer) => buffer.bytes(),
            Self::I8(buffer) => buffer.bytes(),
        }
    }
}
