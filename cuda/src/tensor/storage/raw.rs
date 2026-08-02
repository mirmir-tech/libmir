use mircuda::DeviceBuffer;

use super::{CudaTensor, TensorStorage};
use crate::Result;

impl CudaTensor {
    pub(crate) fn raw_u8(&self) -> Result<DeviceBuffer<u8>> {
        let buffer = match &self.storage {
            TensorStorage::F16(buffer) => buffer.reinterpret()?,
            TensorStorage::Bf16(buffer) => buffer.reinterpret()?,
            TensorStorage::F32(buffer) => buffer.reinterpret()?,
            TensorStorage::F8E4M3(buffer)
            | TensorStorage::F8E5M2(buffer)
            | TensorStorage::U8(buffer) => buffer.clone(),
            TensorStorage::U32(buffer) => buffer.reinterpret()?,
            TensorStorage::I32(buffer) => buffer.reinterpret()?,
            TensorStorage::I8(buffer) => buffer.reinterpret()?,
        };
        Ok(buffer)
    }
}
