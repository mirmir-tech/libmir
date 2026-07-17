use mircuda::{DeviceBuffer, bf16};

use crate::{CudaBackend, CudaTensor, Error, Result};

pub(super) fn norm_buffer(
    backend: &CudaBackend,
    tensor: Option<&CudaTensor>,
    required: bool,
    head_dim: usize,
) -> Result<DeviceBuffer<bf16>> {
    if let Some(tensor) = tensor {
        let buffer = tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: tensor.name().into(),
            expected: "BF16",
        })?;
        if buffer.len() != head_dim {
            return Err(Error::InvalidDecoderKernel("Q/K norm differs from attention head size"));
        }
        return Ok(buffer.clone());
    }
    if required {
        return Err(Error::InvalidDecoderKernel("normalized Q/K projection is missing its weight"));
    }
    Ok(backend.inner.pool.allocate::<bf16>(&backend.inner.stream, head_dim)?)
}
