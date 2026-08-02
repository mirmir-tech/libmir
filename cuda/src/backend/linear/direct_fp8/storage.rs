use mircuda::DeviceBuffer;

use super::CudaBackend;
use crate::{CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result};

pub(super) fn identity_scale(backend: &CudaBackend) -> Result<DeviceBuffer<f32>> {
    let mut host = backend.inner.context.allocate_pinned::<f32>(1)?;
    host.copy_from_slice(&[1.0])?;
    let mut device = backend.inner.pool.allocate::<f32>(&backend.inner.stream, 1)?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

pub(super) fn tensor(
    tensors: &CudaTensorSet,
    name: &str,
    dtype: CudaTensorDType,
    expected: &'static str,
) -> Result<CudaTensor> {
    let tensor = tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?;
    if tensor.dtype() != dtype {
        return Err(Error::DTypeMismatch { name: name.into(), expected });
    }
    Ok(tensor.clone())
}
