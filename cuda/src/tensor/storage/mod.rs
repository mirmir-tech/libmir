use std::collections::HashMap;

use mircuda::{Context, DeviceBuffer, Stream, bf16, f16};

use crate::{Error, Result};

mod raw;
#[cfg(all(test, target_os = "linux"))]
mod tests;

/// Safetensors storage types currently admitted by the CUDA loader.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CudaTensorDType {
    F16,
    Bf16,
    F32,
    /// Raw IEEE-style E4M3 checkpoint values or scaling tensors.
    F8E4M3,
    /// Raw IEEE-style E5M2 checkpoint values.
    F8E5M2,
    U32,
    I32,
    U8,
    I8,
}

impl CudaTensorDType {
    pub(super) fn parse(value: &str) -> Result<Self> {
        match value {
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            "F32" => Ok(Self::F32),
            "F8_E4M3" => Ok(Self::F8E4M3),
            "F8_E5M2" => Ok(Self::F8E5M2),
            "U32" => Ok(Self::U32),
            "I32" => Ok(Self::I32),
            "U8" => Ok(Self::U8),
            "I8" => Ok(Self::I8),
            other => Err(Error::UnsupportedDType(other.into())),
        }
    }

    pub(super) const fn bytes(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F8E4M3 | Self::F8E5M2 | Self::U8 | Self::I8 => 1,
        }
    }
}

/// One device-resident checkpoint tensor.
#[derive(Clone, Debug)]
pub struct CudaTensor {
    pub(super) name: String,
    pub(super) shape: Vec<usize>,
    pub(super) storage: TensorStorage,
}

impl CudaTensor {
    pub(crate) fn from_bf16(name: String, shape: Vec<usize>, buffer: DeviceBuffer<bf16>) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::Bf16(buffer),
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub const fn dtype(&self) -> CudaTensorDType {
        self.storage.dtype()
    }

    #[must_use]
    pub fn bytes(&self) -> usize {
        self.storage.bytes()
    }

    #[must_use]
    pub fn as_bf16(&self) -> Option<&DeviceBuffer<bf16>> {
        match &self.storage {
            TensorStorage::Bf16(buffer) => Some(buffer),
            _ => None,
        }
    }

    /// Returns half-precision storage without copying when the dtype matches.
    #[must_use]
    pub fn as_f16(&self) -> Option<&DeviceBuffer<f16>> {
        match &self.storage {
            TensorStorage::F16(buffer) => Some(buffer),
            _ => None,
        }
    }

    /// Returns packed U32 storage without copying when the dtype matches.
    #[must_use]
    pub fn as_u32(&self) -> Option<&DeviceBuffer<u32>> {
        match &self.storage {
            TensorStorage::U32(buffer) => Some(buffer),
            _ => None,
        }
    }

    /// Returns packed signed I32 storage without copying when the dtype
    /// matches.
    #[must_use]
    pub fn as_i32(&self) -> Option<&DeviceBuffer<i32>> {
        match &self.storage {
            TensorStorage::I32(buffer) => Some(buffer),
            _ => None,
        }
    }

    /// Returns byte storage without copying when the dtype is exactly U8.
    #[must_use]
    pub fn as_u8(&self) -> Option<&DeviceBuffer<u8>> {
        match &self.storage {
            TensorStorage::U8(buffer) => Some(buffer),
            _ => None,
        }
    }

    /// Returns raw E4M3 storage without converting scale values on the host.
    #[must_use]
    pub fn as_f8_e4m3(&self) -> Option<&DeviceBuffer<u8>> {
        match &self.storage {
            TensorStorage::F8E4M3(buffer) => Some(buffer),
            _ => None,
        }
    }

    /// Returns raw E5M2 storage without converting values on the host.
    #[must_use]
    pub fn as_f8_e5m2(&self) -> Option<&DeviceBuffer<u8>> {
        match &self.storage {
            TensorStorage::F8E5M2(buffer) => Some(buffer),
            _ => None,
        }
    }

    /// Returns single-precision storage without copying when the dtype matches.
    #[must_use]
    pub fn as_f32(&self) -> Option<&DeviceBuffer<f32>> {
        match &self.storage {
            TensorStorage::F32(buffer) => Some(buffer),
            _ => None,
        }
    }
}

/// Completed device tensor collection sharing one explicit CUDA stream.
///
/// Cloning the set retains shared allocation handles without copying tensors.
#[derive(Clone, Debug)]
pub struct CudaTensorSet {
    pub(super) tensors: HashMap<String, CudaTensor>,
    pub(super) context: Context,
    pub(super) stream: Stream,
}

impl CudaTensorSet {
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&CudaTensor> {
        self.tensors
            .get(name)
            .or_else(|| self.tensors.get(&models::weights::alternate_model_tensor_name(name)))
    }

    /// Explicit diagnostic readback of one BF16 tensor.
    pub fn read_bf16(&self, name: &str) -> Result<Vec<bf16>> {
        let tensor = self.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?;
        let device = tensor
            .as_bf16()
            .ok_or_else(|| Error::DTypeMismatch { name: name.into(), expected: "BF16" })?;
        let mut host = self.context.allocate_pinned::<bf16>(device.len())?;
        self.stream.copy_to_host(device, &mut host)?;
        Ok(host.to_vec()?)
    }
}

#[derive(Clone, Debug)]
pub(super) enum TensorStorage {
    F16(DeviceBuffer<f16>),
    Bf16(DeviceBuffer<bf16>),
    F32(DeviceBuffer<f32>),
    F8E4M3(DeviceBuffer<u8>),
    F8E5M2(DeviceBuffer<u8>),
    U32(DeviceBuffer<u32>),
    I32(DeviceBuffer<i32>),
    U8(DeviceBuffer<u8>),
    I8(DeviceBuffer<i8>),
}

impl TensorStorage {
    const fn dtype(&self) -> CudaTensorDType {
        match self {
            Self::F16(_) => CudaTensorDType::F16,
            Self::Bf16(_) => CudaTensorDType::Bf16,
            Self::F32(_) => CudaTensorDType::F32,
            Self::F8E4M3(_) => CudaTensorDType::F8E4M3,
            Self::F8E5M2(_) => CudaTensorDType::F8E5M2,
            Self::U32(_) => CudaTensorDType::U32,
            Self::I32(_) => CudaTensorDType::I32,
            Self::U8(_) => CudaTensorDType::U8,
            Self::I8(_) => CudaTensorDType::I8,
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Self::F16(buffer) => buffer.bytes(),
            Self::Bf16(buffer) => buffer.bytes(),
            Self::F32(buffer) => buffer.bytes(),
            Self::F8E4M3(buffer) | Self::F8E5M2(buffer) | Self::U8(buffer) => buffer.bytes(),
            Self::U32(buffer) => buffer.bytes(),
            Self::I32(buffer) => buffer.bytes(),
            Self::I8(buffer) => buffer.bytes(),
        }
    }
}
