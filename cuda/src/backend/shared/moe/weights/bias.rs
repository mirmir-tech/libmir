//! The selection bias of a sigmoid router, kept in FP32 whatever dtype the
//! checkpoint stores it in.

use mircuda::DeviceBuffer;
use models::weights::TensorBinding;

use crate::{CudaBackend, CudaTensorSet, Error, Result};

/// The bias as FP32; a BF16 one is widened once at load.
pub(super) fn correction_bias(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    binding: &TensorBinding,
) -> Result<DeviceBuffer<f32>> {
    let tensor = tensors
        .get(&binding.source)
        .ok_or_else(|| Error::MissingTensor(binding.source.clone()))?;
    if let Some(values) = tensor.as_f32() {
        return Ok(values.clone());
    }
    let values = tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: binding.source.clone(),
        expected: "F32",
    })?;
    let widened = backend.read_logits(values)?;
    let mut staging = backend.inner.context.allocate_pinned(widened.len())?;
    staging.copy_from_slice(&widened)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, widened.len())?;
    backend.inner.stream.copy_to_device(&mut staging, &mut device)?;
    Ok(device)
}
