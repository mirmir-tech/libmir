use models::weights::{
    GptqBits, GptqCheckpointFormat, GptqScaleDType, TensorBinding, TensorStorage,
};

use crate::engine::{Array, Dtype, Error, ModelTensors, Result, Stream, linear::GptqLinear};

pub(super) fn linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<GptqLinear> {
    let TensorStorage::Gptq {
        format,
        scales,
        zero_points,
        group_indices,
    } = &binding.storage
    else {
        return Err(invalid(binding, "binding is not GPTQ"));
    };
    if format.bits != GptqBits::Four
        || format.scale_dtype != GptqScaleDType::F16
        || !format.symmetric
        || !format.activation_order
        || !format.is_input_packed()
    {
        return Err(invalid(binding, "format is not activation-ordered Metal GPTQ W4A16"));
    }
    let logical = binding
        .logical_shape
        .as_deref()
        .ok_or_else(|| invalid(binding, "logical shape is missing"))?;
    let [output, input] = logical else {
        return Err(invalid(binding, "logical shape is not a matrix"));
    };
    if format.group_size == 0 || !input.is_multiple_of(format.group_size) {
        return Err(invalid(binding, "GPTQ group geometry is invalid"));
    }
    let groups = input / format.group_size;
    let graph = stream.native().graph();
    let weight = tensors.get(&binding.source)?;
    let zero_points = tensors.get(zero_points)?;
    let scales = tensors.get(scales)?;
    let group_indices = tensors.get(group_indices)?;
    require(&weight, Dtype::Int32, &[*input / 8, *output], binding)?;
    require(&zero_points, Dtype::Int32, &[groups, *output / 8], binding)?;
    require(&scales, Dtype::Float16, &[groups, *output], binding)?;
    require(&group_indices, Dtype::Int32, &[*input], binding)?;
    let weight = Array::from_native(graph.view_dtype(weight.native(), mirtal::DType::Uint32)?)?;
    let zeros = Array::from_native(graph.view_dtype(zero_points.native(), mirtal::DType::Uint32)?)?;
    Ok(GptqLinear::new(
        [weight, zeros, scales, group_indices],
        *input,
        *output,
        format.group_size,
        format.checkpoint_format == GptqCheckpointFormat::Gptq,
    ))
}

fn require(array: &Array, dtype: Dtype, shape: &[usize], binding: &TensorBinding) -> Result<()> {
    let expected = shape
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if array.dtype()? == dtype && array.shape()? == expected {
        Ok(())
    } else {
        Err(invalid(binding, "physical dtype or shape differs from the GPTQ contract"))
    }
}

fn invalid(binding: &TensorBinding, reason: &str) -> Error {
    Error::InvalidQuantization(format!("{}: {reason}", binding.source))
}
