use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use mircuda::{Context, DeviceElement, MemoryPool, Stream, bf16, f16};
use models::weights::TensorInfo;

use self::pending::{PendingBuffer, PendingStorage, PendingTensor};
use super::storage::{CudaTensor, CudaTensorDType, CudaTensorSet};
use crate::{Error, Result, kernels::DenseCast};

mod pending;

const DEFAULT_STAGING_LIMIT: usize = 512 * 1024 * 1024;

/// Asynchronous checkpoint upload batch with bounded pinned staging.
#[derive(Debug)]
pub struct TensorUploadBatch {
    context: Context,
    stream: Stream,
    pool: MemoryPool,
    pending: Vec<PendingTensor>,
    tensors: HashMap<String, CudaTensor>,
    staging_bytes: usize,
    staging_limit: usize,
}

impl TensorUploadBatch {
    pub(crate) fn new(context: Context, stream: Stream, pool: MemoryPool) -> Self {
        Self {
            context,
            stream,
            pool,
            pending: Vec::new(),
            tensors: HashMap::new(),
            staging_bytes: 0,
            staging_limit: DEFAULT_STAGING_LIMIT,
        }
    }

    /// Reads one payload directly into pinned memory and enqueues one H2D
    /// transfer.
    pub fn enqueue(&mut self, info: &TensorInfo) -> Result<()> {
        let (dtype, elements) = metadata(info)?;
        let storage = match dtype {
            CudaTensorDType::F16 => PendingStorage::F16(self.stage::<f16>(info, elements)?),
            CudaTensorDType::Bf16 => PendingStorage::Bf16(self.stage::<bf16>(info, elements)?),
            CudaTensorDType::F32 => PendingStorage::F32(self.stage::<f32>(info, elements)?),
            CudaTensorDType::F8E4M3 => PendingStorage::F8E4M3(self.stage::<u8>(info, elements)?),
            CudaTensorDType::F8E5M2 => PendingStorage::F8E5M2(self.stage::<u8>(info, elements)?),
            CudaTensorDType::U32 => PendingStorage::U32(self.stage::<u32>(info, elements)?),
            CudaTensorDType::I32 => PendingStorage::I32(self.stage::<i32>(info, elements)?),
            CudaTensorDType::U8 => PendingStorage::U8(self.stage::<u8>(info, elements)?),
            CudaTensorDType::I8 => PendingStorage::I8(self.stage::<i8>(info, elements)?),
        };
        self.push(info, storage)
    }

    pub(crate) fn enqueue_as_bf16(&mut self, info: &TensorInfo, cast: &DenseCast) -> Result<()> {
        let (dtype, elements) = metadata(info)?;
        let storage = match dtype {
            CudaTensorDType::Bf16 => PendingStorage::Bf16(self.stage::<bf16>(info, elements)?),
            CudaTensorDType::F16 => {
                let source = self.stage::<f16>(info, elements)?;
                let mut output = self.pool.allocate::<bf16>(&self.stream, elements)?;
                cast.f16_to_bf16(&self.stream, &source.device, &mut output)?;
                PendingStorage::F16ToBf16 { source, output }
            },
            CudaTensorDType::F32 => {
                let source = self.stage::<f32>(info, elements)?;
                let mut output = self.pool.allocate::<bf16>(&self.stream, elements)?;
                cast.f32_to_bf16(&self.stream, &source.device, &mut output)?;
                PendingStorage::F32ToBf16 { source, output }
            },
            _ => {
                return Err(Error::DTypeMismatch {
                    name: info.name.clone(),
                    expected: "F16, BF16, or F32 dense storage",
                });
            },
        };
        self.push(info, storage)
    }

    pub(crate) fn enqueue_float_as_bf16(
        &mut self,
        info: &TensorInfo,
        cast: &DenseCast,
    ) -> Result<()> {
        match CudaTensorDType::parse(&info.dtype)? {
            CudaTensorDType::F16 | CudaTensorDType::Bf16 | CudaTensorDType::F32 => {
                self.enqueue_as_bf16(info, cast)
            },
            _ => self.enqueue(info),
        }
    }

    /// Synchronizes the final bounded batch and returns device tensors.
    pub fn finish(mut self) -> Result<CudaTensorSet> {
        self.flush()?;
        Ok(CudaTensorSet {
            tensors: self.tensors,
            context: self.context,
            stream: self.stream,
        })
    }

    fn push(&mut self, info: &TensorInfo, storage: PendingStorage) -> Result<()> {
        self.staging_bytes =
            self.staging_bytes.checked_add(info.payload_bytes()?).ok_or_else(|| {
                Error::InvalidTensorSize {
                    name: info.name.clone(),
                    expected: usize::MAX,
                    actual: self.staging_bytes,
                }
            })?;
        self.pending
            .push(PendingTensor::new(info.name.clone(), info.shape.clone(), storage));
        if self.staging_bytes >= self.staging_limit {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.stream.synchronize()?;
        for pending in self.pending.drain(..) {
            let tensor = pending.complete();
            let name = tensor.name.clone();
            if self.tensors.insert(name.clone(), tensor).is_some() {
                return Err(Error::DuplicateTensor(name));
            }
        }
        self.staging_bytes = 0;
        Ok(())
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn set_staging_limit(&mut self, bytes: usize) {
        self.staging_limit = bytes.max(1);
    }

    fn stage<T: DeviceElement>(
        &self,
        info: &TensorInfo,
        elements: usize,
    ) -> Result<PendingBuffer<T>> {
        let offset = info.payload_start()?;
        let length = info.payload_bytes()?;
        let mut file = File::open(&info.file)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut staging = self.context.allocate_pinned::<T>(elements)?;
        let read = staging.with_bytes_mut(|bytes| file.read_exact(bytes))?;
        read?;
        release_file_cache(&file, offset, length);
        let mut device = self.pool.allocate::<T>(&self.stream, elements)?;
        self.stream.copy_to_device(&mut staging, &mut device)?;
        Ok(PendingBuffer { device, staging })
    }
}

#[cfg(target_os = "linux")]
fn release_file_cache(file: &File, offset: u64, length: usize) {
    let length = u64::try_from(length).unwrap_or(u64::MAX);
    if let Err(error) = rustix::fs::fadvise(
        file,
        offset,
        std::num::NonZeroU64::new(length),
        rustix::fs::Advice::DontNeed,
    ) {
        tracing::debug!(%error, offset, length, "could not release checkpoint file cache");
    }
}

#[cfg(not(target_os = "linux"))]
fn release_file_cache(_file: &File, _offset: u64, _length: usize) {}

fn metadata(info: &TensorInfo) -> Result<(CudaTensorDType, usize)> {
    let dtype = CudaTensorDType::parse(&info.dtype)?;
    let elements = element_count(&info.shape, &info.name)?;
    let actual = info.payload_bytes()?;
    let expected = elements.checked_mul(dtype.bytes()).ok_or_else(|| Error::InvalidTensorSize {
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
    Ok((dtype, elements))
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
