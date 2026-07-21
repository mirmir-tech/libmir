use models::weights::{TensorBinding, TensorStorage};

use super::{Error, ModelTensors, NormWeight, QuantizedEmbedding, QuantizedLinear, Result, Stream};

pub(super) fn affine_linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
) -> Result<QuantizedLinear> {
    let TensorStorage::AffineQuantized {
        scales,
        biases: Some(biases),
        output_bias,
        group_size: Some(group_size),
        ..
    } = &binding.storage
    else {
        return Err(invalid("linear", binding));
    };
    QuantizedLinear::load_names(
        tensors,
        &binding.source,
        scales,
        biases,
        output_bias.as_deref(),
        i32::try_from(*group_size)?,
    )
}

pub(super) fn affine_embedding(
    tensors: &ModelTensors,
    binding: &TensorBinding,
) -> Result<QuantizedEmbedding> {
    let TensorStorage::AffineQuantized {
        scales,
        biases: Some(biases),
        group_size: Some(group_size),
        ..
    } = &binding.storage
    else {
        return Err(invalid("embedding", binding));
    };
    QuantizedEmbedding::load_names(
        tensors,
        &binding.source,
        scales,
        biases,
        i32::try_from(*group_size)?,
    )
}

pub(super) fn adjusted_norm(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    shift: f32,
    stream: &Stream,
) -> Result<NormWeight> {
    NormWeight::load_name_adjusted(tensors, &binding.source, shift, stream)
}

fn invalid(kind: &str, binding: &TensorBinding) -> Error {
    Error::InvalidQuantization(format!(
        "{kind} requires a complete affine binding: {}",
        binding.source
    ))
}
