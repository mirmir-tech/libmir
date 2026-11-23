use mircuda::{DeviceBuffer, bf16};
use models::weights::TensorStorage;

use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    checkpoint::dense::DenseLayerBindings, kernels::RopeRows,
};

/// Restores Hugging Face's `rotate_half` row order in the layer's query and
/// key projections, which the checkpoint declares interleaved pairwise
/// (llama.cpp's GGUF files of Llama-family architectures). Only the first
/// `rotary_dim` rows of a head rotate (GLM-4); a projection bias moves with
/// its rows. Dense and GGML rows move as bytes, before any format loader
/// reads them.
pub(super) fn restore(
    backend: &CudaBackend,
    tensors: &mut CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
    dims: (usize, usize),
) -> Result<()> {
    let operation = RopeRows::compile(backend.compiler())?;
    for binding in [bindings.attention.query, bindings.attention.key] {
        let (TensorStorage::Dense { bias, .. } | TensorStorage::Ggml { bias, .. }) =
            &binding.storage
        else {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "interleaved rotary rows need a dense or GGML projection: {}",
                binding.source
            )));
        };
        let rows = binding
            .logical_shape
            .as_deref()
            .unwrap_or(&binding.shape)
            .first()
            .copied()
            .ok_or_else(|| Error::MissingTensor(binding.source.clone()))?;
        for name in std::iter::once(&binding.source).chain(bias) {
            restore_rows(backend, &operation, tensors, name, (rows, dims))?;
        }
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
    let source = match tensor.dtype() {
        CudaTensorDType::U8 => tensor.as_u8().cloned(),
        CudaTensorDType::Bf16 => tensor.as_bf16().map(DeviceBuffer::reinterpret).transpose()?,
        _ => None,
    }
    .ok_or_else(|| Error::DTypeMismatch {
        name: name.into(),
        expected: "BF16 or GGML blocks",
    })?;
    let mut output = backend.pool().allocate::<u8>(backend.stream(), source.len())?;
    operation.execute(backend.stream(), &source, &mut output, rows, dims)?;
    let (name, shape) = (tensor.name().to_owned(), tensor.shape().to_vec());
    tensors.replace(if tensor.dtype() == CudaTensorDType::U8 {
        CudaTensor::from_u8(name, shape, output)
    } else {
        CudaTensor::from_bf16(name, shape, output.reinterpret::<bf16>()?)
    })
}
