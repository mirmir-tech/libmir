use mircuda::{DeviceBuffer, DeviceElement, PinnedBuffer, bf16, f16};

use super::super::storage::{CudaTensor, TensorStorage};

#[derive(Debug)]
pub(super) struct PendingTensor {
    name: String,
    shape: Vec<usize>,
    storage: PendingStorage,
}

impl PendingTensor {
    pub(super) const fn new(name: String, shape: Vec<usize>, storage: PendingStorage) -> Self {
        Self { name, shape, storage }
    }

    pub(super) fn complete(self) -> CudaTensor {
        CudaTensor {
            name: self.name,
            shape: self.shape,
            storage: self.storage.complete(),
        }
    }
}

#[derive(Debug)]
pub(super) struct PendingBuffer<T: DeviceElement> {
    pub(super) device: DeviceBuffer<T>,
    pub(super) staging: PinnedBuffer<T>,
}

#[derive(Debug)]
pub(super) enum PendingStorage {
    F16(PendingBuffer<f16>),
    Bf16(PendingBuffer<bf16>),
    F32(PendingBuffer<f32>),
    F8E4M3(PendingBuffer<u8>),
    F8E5M2(PendingBuffer<u8>),
    U32(PendingBuffer<u32>),
    I32(PendingBuffer<i32>),
    U8(PendingBuffer<u8>),
    I8(PendingBuffer<i8>),
    F16ToBf16 {
        source: PendingBuffer<f16>,
        output: DeviceBuffer<bf16>,
    },
    F32ToBf16 {
        source: PendingBuffer<f32>,
        output: DeviceBuffer<bf16>,
    },
}

impl PendingStorage {
    fn complete(self) -> TensorStorage {
        match self {
            Self::F16(PendingBuffer { device, staging: _staging }) => TensorStorage::F16(device),
            Self::Bf16(PendingBuffer { device, staging: _staging }) => TensorStorage::Bf16(device),
            Self::F32(PendingBuffer { device, staging: _staging }) => TensorStorage::F32(device),
            Self::F8E4M3(PendingBuffer { device, staging: _staging }) => {
                TensorStorage::F8E4M3(device)
            },
            Self::F8E5M2(PendingBuffer { device, staging: _staging }) => {
                TensorStorage::F8E5M2(device)
            },
            Self::U32(PendingBuffer { device, staging: _staging }) => TensorStorage::U32(device),
            Self::I32(PendingBuffer { device, staging: _staging }) => TensorStorage::I32(device),
            Self::U8(PendingBuffer { device, staging: _staging }) => TensorStorage::U8(device),
            Self::I8(PendingBuffer { device, staging: _staging }) => TensorStorage::I8(device),
            Self::F16ToBf16 { source: _source, output } => TensorStorage::Bf16(output),
            Self::F32ToBf16 { source: _source, output } => TensorStorage::Bf16(output),
        }
    }
}
