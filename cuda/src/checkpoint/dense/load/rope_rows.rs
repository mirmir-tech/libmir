use mircuda::bf16;
use models::weights::{DenseDecoderLayerBindings, TensorStorage};

use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result, kernels::RopeRows,
};

/// Restores Hugging Face's `rotate_half` row order in the layer's query and
/// key projections, which the checkpoint declares interleaved pairwise
/// (llama.cpp's GGUF files of Llama-family architectures). Dense and GGML
/// rows move as bytes, before any format loader reads them.
pub(super) fn restore(
    backend: &CudaBackend,
    tensors: &mut CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
    head_dim: usize,
) -> Result<()> {
    let operation = RopeRows::compile(backend.compiler())?;
    for binding in [bindings.attention.query, bindings.attention.key] {
        let plain = matches!(
            binding.storage,
            TensorStorage::Dense { bias: None, .. } | TensorStorage::Ggml { bias: None, .. }
        );
        let rows = binding.logical_shape.as_deref().unwrap_or(&binding.shape).first().copied();
        let (true, Some(rows)) = (plain, rows) else {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "interleaved rotary rows need a dense or GGML projection without bias: {}",
                binding.source
            )));
        };
        let tensor = tensors
            .get(&binding.source)
            .cloned()
            .ok_or_else(|| Error::MissingTensor(binding.source.clone()))?;
        let source = match tensor.dtype() {
            CudaTensorDType::U8 => tensor.as_u8().cloned(),
            CudaTensorDType::Bf16 => {
                tensor.as_bf16().map(mircuda::DeviceBuffer::reinterpret).transpose()?
            },
            _ => None,
        }
        .ok_or_else(|| Error::DTypeMismatch {
            name: binding.source.clone(),
            expected: "BF16 or GGML blocks",
        })?;
        let mut output = backend.pool().allocate::<u8>(backend.stream(), source.len())?;
        operation.execute(backend.stream(), &source, &mut output, rows, head_dim)?;
        let (name, shape) = (tensor.name().to_owned(), tensor.shape().to_vec());
        tensors.replace(if tensor.dtype() == CudaTensorDType::U8 {
            CudaTensor::from_u8(name, shape, output)
        } else {
            CudaTensor::from_bf16(name, shape, output.reinterpret::<bf16>()?)
        })?;
    }
    Ok(())
}
