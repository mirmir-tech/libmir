use mircuda::DeviceBuffer;

use super::{CudaTensor, TensorStorage};

impl CudaTensor {
    pub(crate) fn from_u8(name: String, shape: Vec<usize>, buffer: DeviceBuffer<u8>) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::U8(buffer),
        }
    }

    pub(crate) fn from_u32(name: String, shape: Vec<usize>, buffer: DeviceBuffer<u32>) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::U32(buffer),
        }
    }

    pub(crate) fn from_f8_e4m3(name: String, shape: Vec<usize>, buffer: DeviceBuffer<u8>) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::F8E4M3(buffer),
        }
    }

    pub(crate) fn from_f8_e5m2(name: String, shape: Vec<usize>, buffer: DeviceBuffer<u8>) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::F8E5M2(buffer),
        }
    }

    pub(crate) fn from_f32(name: String, shape: Vec<usize>, buffer: DeviceBuffer<f32>) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::F32(buffer),
        }
    }
}
