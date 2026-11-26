use mircuda::bf16;
use models::weights::TensorStorage;

use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, Error, Result,
    checkpoint::dense::DenseLayerBindings,
    kernels::{GgmlBlocks, RopeRows},
};

/// Restores Hugging Face's `rotate_half` row order in the layer's query and
/// key projections, which the checkpoint declares interleaved pairwise
/// (llama.cpp's GGUF files of Llama-family architectures). Only the first
/// `rotary_dim` rows of a head rotate (GLM-4); a projection bias moves with
/// its rows, and a learned query or key norm (Cohere's per-head norms, or
/// one shared by every head) with the features of each head it scales.
/// Dense and GGML rows move as bytes, before any format loader reads them.
pub(super) fn restore(
    backend: &CudaBackend,
    tensors: &mut CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
    dims: (usize, usize),
) -> Result<()> {
    let operation = RopeRows::compile(backend.compiler())?;
    let key = match bindings.attention {
        crate::checkpoint::dense::DenseAttentionBindings::Projected(attention) => {
            Some(attention.key)
        },
        crate::checkpoint::dense::DenseAttentionBindings::Shared { .. } => None,
    };
    for binding in std::iter::once(bindings.attention.query()).chain(key) {
        let bias = match &binding.storage {
            TensorStorage::Dense { bias, .. } => bias.as_deref(),
            TensorStorage::Ggml { companion, .. } => companion.bias(),
            _ => {
                return Err(Error::UnsupportedDecoderLayer(format!(
                    "interleaved rotary rows need a dense or GGML projection: {}",
                    binding.source
                )));
            },
        };
        let rows = binding
            .logical_shape
            .as_deref()
            .unwrap_or(&binding.shape)
            .first()
            .copied()
            .ok_or_else(|| Error::MissingTensor(binding.source.clone()))?;
        for name in std::iter::once(binding.source.as_str()).chain(bias) {
            restore_rows(backend, &operation, tensors, name, (rows, dims))?;
        }
    }
    // A norm weight holds one value per feature of a head, or of every head;
    // one tensor bound as both norms moves once.
    let query = bindings.attention.query_norm();
    let key = bindings.attention.key_norm();
    let key = key.filter(|key| query.is_none_or(|query| query.source != key.source));
    for norm in query.into_iter().chain(key) {
        let features = norm.shape.iter().product();
        restore_rows(backend, &operation, tensors, &norm.source, (features, dims))?;
    }
    Ok(())
}

/// One tensor of `rows` leading rows in restored order.
fn restore_rows(
    backend: &CudaBackend,
    operation: &RopeRows,
    tensors: &mut CudaTensorSet,
    name: &str,
    (rows, dims): (usize, (usize, usize)),
) -> Result<()> {
    let tensor = tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))?;
    let (name, shape) = (tensor.name().to_owned(), tensor.shape().to_vec());
    let (stream, pool) = (backend.stream(), backend.pool());
    // GGML blocks move into a padded allocation, as uploaded; BF16 values
    // into a plain one.
    let restored = if let Some(blocks) = tensor.as_ggml_blocks() {
        let mut output = GgmlBlocks::allocate(pool, stream, blocks.len())?;
        operation.execute(stream, blocks.view(), output.view_mut(), rows, dims)?;
        CudaTensor::from_ggml_blocks(name, shape, output)
    } else if let Some(values) = tensor.as_bf16() {
        let source = values.reinterpret::<u8>()?;
        let mut output = pool.allocate::<u8>(stream, source.len())?;
        operation.execute(stream, &source, &mut output, rows, dims)?;
        CudaTensor::from_bf16(name, shape, output.reinterpret::<bf16>()?)
    } else {
        return Err(Error::DTypeMismatch { name, expected: "BF16 or GGML blocks" });
    };
    tensors.replace(restored)
}

#[cfg(test)]
mod tests;
