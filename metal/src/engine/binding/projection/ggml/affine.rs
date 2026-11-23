//! Dense GGML projections MLX's affine products run: types whose 32-value
//! sub-blocks share one scale and minimum, repacked once at load. Banks,
//! scaled matrices and other types keep the GGML kernels.

use models::weights::{TensorBinding, TensorStorage};

use super::linear;
use crate::engine::{
    ModelTensors, QuantizedArrays, QuantizedLinear, Result, Stream, kernels::ggml::AffineLayout,
};

/// The projection as MLX affine groups of 32 with bf16 scales, when its type
/// and shape allow it.
pub(in crate::engine::binding::projection) fn linear_affine(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<Option<QuantizedLinear>> {
    let TensorStorage::Ggml { format, scale: None, .. } = &binding.storage else {
        return Ok(None);
    };
    let Some(layout) = AffineLayout::of(format.ggml_type) else {
        return Ok(None);
    };
    let matrix = linear(tensors, binding, stream)?;
    if matrix.experts.is_some() {
        return Ok(None);
    }
    let [weight, scales, biases] = stream.kernels().ggml_affine(
        stream,
        &matrix.weight,
        matrix.matrix,
        layout,
        mirtal::DType::Bfloat16,
    )?;
    let group_size = AffineLayout::GROUP_SIZE;
    let arrays = QuantizedArrays::new(weight, scales, biases, group_size, layout.bits())?;
    Ok(Some(
        QuantizedLinear::from_quantized(arrays, group_size, layout.bits()).with_bias(matrix.bias),
    ))
}
