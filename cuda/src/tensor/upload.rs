use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use mircuda::{Context, DeviceBuffer, DeviceElement, MemoryPool, PinnedBuffer, Stream, bf16, f16};
use models::weights::TensorInfo;

use super::storage::{CudaTensor, CudaTensorDType, CudaTensorSet, TensorStorage};
use crate::{Error, Result};

/// Asynchronous checkpoint upload batch retaining pinned staging allocations.
#[derive(Debug)]
pub struct TensorUploadBatch {
    context: Context,
    stream: Stream,
    pool: MemoryPool,
    pending: Vec<PendingTensor>,
}

impl TensorUploadBatch {
    pub(crate) const fn new(context: Context, stream: Stream, pool: MemoryPool) -> Self {
        Self {
            context,
            stream,
            pool,
            pending: Vec::new(),
        }
    }

    /// Reads one payload directly into pinned memory and enqueues one H2D
    /// transfer.
    pub fn enqueue(&mut self, info: &TensorInfo) -> Result<()> {
        let dtype = CudaTensorDType::parse(&info.dtype)?;
        let elements = element_count(&info.shape, &info.name)?;
        let actual = info.payload_bytes()?;
        let expected =
            elements.checked_mul(dtype.bytes()).ok_or_else(|| Error::InvalidTensorSize {
                name: info.name.clone(),
                expected: usize::MAX,
                actual,
            })?;
        if expected != actual {
            return Err(Error::InvalidTensorSize {
                name: info.name.clone(),
                expected,
                actual,
            });
        }
        let storage = match dtype {
            CudaTensorDType::F16 => PendingStorage::F16(self.stage::<f16>(info, elements)?),
            CudaTensorDType::Bf16 => PendingStorage::Bf16(self.stage::<bf16>(info, elements)?),
            CudaTensorDType::F32 => PendingStorage::F32(self.stage::<f32>(info, elements)?),
            CudaTensorDType::F8E4M3 => PendingStorage::F8E4M3(self.stage::<u8>(info, elements)?),
            CudaTensorDType::U32 => PendingStorage::U32(self.stage::<u32>(info, elements)?),
            CudaTensorDType::I32 => PendingStorage::I32(self.stage::<i32>(info, elements)?),
            CudaTensorDType::U8 => PendingStorage::U8(self.stage::<u8>(info, elements)?),
            CudaTensorDType::I8 => PendingStorage::I8(self.stage::<i8>(info, elements)?),
        };
        self.pending.push(PendingTensor {
            name: info.name.clone(),
            shape: info.shape.clone(),
            storage,
        });
        Ok(())
    }

    /// Synchronizes the batch once, releases staging, and returns device
    /// tensors.
    pub fn finish(self) -> Result<CudaTensorSet> {
        self.stream.synchronize()?;
        let mut tensors = HashMap::with_capacity(self.pending.len());
        for pending in self.pending {
            let tensor = pending.complete();
            let name = tensor.name.clone();
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(Error::DuplicateTensor(name));
            }
        }
        Ok(CudaTensorSet {
            tensors,
            context: self.context,
            stream: self.stream,
        })
    }

    fn stage<T: DeviceElement>(
        &self,
        info: &TensorInfo,
        elements: usize,
    ) -> Result<PendingBuffer<T>> {
        let mut file = File::open(&info.file)?;
        file.seek(SeekFrom::Start(info.payload_start()?))?;
        let mut staging = self.context.allocate_pinned::<T>(elements)?;
        let read = staging.with_bytes_mut(|bytes| file.read_exact(bytes))?;
        read?;
        let mut device = self.pool.allocate::<T>(&self.stream, elements)?;
        self.stream.copy_to_device(&mut staging, &mut device)?;
        Ok(PendingBuffer { device, staging })
    }
}

#[derive(Debug)]
struct PendingTensor {
    name: String,
    shape: Vec<usize>,
    storage: PendingStorage,
}

impl PendingTensor {
    fn complete(self) -> CudaTensor {
        CudaTensor {
            name: self.name,
            shape: self.shape,
            storage: self.storage.complete(),
        }
    }
}

#[derive(Debug)]
struct PendingBuffer<T: DeviceElement> {
    device: DeviceBuffer<T>,
    staging: PinnedBuffer<T>,
}

#[derive(Debug)]
enum PendingStorage {
    F16(PendingBuffer<f16>),
    Bf16(PendingBuffer<bf16>),
    F32(PendingBuffer<f32>),
    F8E4M3(PendingBuffer<u8>),
    U32(PendingBuffer<u32>),
    I32(PendingBuffer<i32>),
    U8(PendingBuffer<u8>),
    I8(PendingBuffer<i8>),
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
            Self::U32(PendingBuffer { device, staging: _staging }) => TensorStorage::U32(device),
            Self::I32(PendingBuffer { device, staging: _staging }) => TensorStorage::I32(device),
            Self::U8(PendingBuffer { device, staging: _staging }) => TensorStorage::U8(device),
            Self::I8(PendingBuffer { device, staging: _staging }) => TensorStorage::I8(device),
        }
    }
}

fn element_count(shape: &[usize], name: &str) -> Result<usize> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| Error::InvalidTensorSize {
            name: name.into(),
            expected: usize::MAX,
            actual: 0,
        })
    })
}
