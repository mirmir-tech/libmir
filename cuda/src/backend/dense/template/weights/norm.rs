use mircuda::{DeviceBuffer, bf16};

use super::DenseWeightSource;
use crate::{
    CudaBackend, CudaTensor, DecodeAttentionConfig, Error, Result, kernels::QkNormWeights,
};

/// The query and key norm weights, sized for the norm's scope.
pub(super) fn qk_norms(
    backend: &CudaBackend,
    attention: DecodeAttentionConfig,
    source: DenseWeightSource<'_>,
) -> Result<[DeviceBuffer<bf16>; 2]> {
    let normalization = attention.qkv_normalization;
    let [query, key] = normalization.weight_lengths(
        attention.query_heads,
        attention.cache.kv_heads,
        attention.cache.key_head_dim,
    );
    if normalization.weights == QkNormWeights::Unit {
        if source.query_norm.is_some() || source.key_norm.is_some() {
            return Err(Error::InvalidDecoderKernel("unit Q/K norms have a learned weight"));
        }
        return Ok([unit_buffer(backend, query)?, unit_buffer(backend, key)?]);
    }
    Ok([
        norm_buffer(backend, source.query_norm, normalization.query, query)?,
        norm_buffer(backend, source.key_norm, normalization.key, key)?,
    ])
}

fn unit_buffer(backend: &CudaBackend, length: usize) -> Result<DeviceBuffer<bf16>> {
    let mut host = backend.inner.context.allocate_pinned::<bf16>(length)?;
    host.copy_from_slice(&vec![bf16::ONE; length])?;
    let mut device = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, length)?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    // The pinned staging buffer is released on return.
    backend.inner.stream.synchronize()?;
    Ok(device)
}

fn norm_buffer(
    backend: &CudaBackend,
    tensor: Option<&CudaTensor>,
    required: bool,
    length: usize,
) -> Result<DeviceBuffer<bf16>> {
    if let Some(tensor) = tensor {
        let buffer = tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: tensor.name().into(),
            expected: "BF16",
        })?;
        if buffer.len() != length {
            return Err(Error::InvalidDecoderKernel("Q/K norm differs from its normalized width"));
        }
        return Ok(buffer.clone());
    }
    if required {
        return Err(Error::InvalidDecoderKernel("normalized Q/K projection is missing its weight"));
    }
    Ok(backend.inner.pool.allocate::<bf16>(&backend.inner.stream, length)?)
}
