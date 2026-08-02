use models::weights::{
    CompressedIntegerScaleDType, CompressedIntegerScaleStrategy, GptqBits, GptqCheckpointFormat,
    GptqScaleDType, TensorBinding, TensorStorage,
};

use crate::engine::{
    Array, Dtype, Error, ModelTensors, QuantizedArrays, QuantizedEmbedding, QuantizedLinear,
    Result, Stream,
};

const NATIVE_INT8_GROUP_SIZE: usize = 64;

pub(super) fn linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<QuantizedLinear> {
    let (arrays, group_size, bits) = arrays(tensors, binding, stream)?;
    Ok(QuantizedLinear::from_quantized(arrays, group_size, bits))
}

pub(super) fn embedding(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<QuantizedEmbedding> {
    let (arrays, group_size, bits) = arrays(tensors, binding, stream)?;
    Ok(QuantizedEmbedding::from_quantized(arrays, group_size, bits))
}

pub(super) fn awq_linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<QuantizedLinear> {
    let TensorStorage::Awq { format, scales, zero_points } = &binding.storage else {
        return Err(invalid(binding, "binding is not AWQ"));
    };
    if !format.is_gemm_w4a16() {
        return Err(invalid(binding, "format is not AWQ GEMM W4A16"));
    }
    let [output, input] = matrix_shape(binding)?;
    let groups = input / format.group_size;
    let packed_output = output / 8;
    let graph = stream.native().graph();
    let weight = tensors.get(&binding.source)?;
    let zero_points = tensors.get(zero_points)?;
    let scales = tensors.get(scales)?;
    require(&weight, Dtype::Int32, &[input, packed_output], binding)?;
    require(&zero_points, Dtype::Int32, &[groups, packed_output], binding)?;
    require(&scales, Dtype::Float16, &[groups, output], binding)?;
    let weight = Array::from_native(graph.view_dtype(weight.native(), mirtal::DType::Uint32)?)?;
    let zero_points =
        Array::from_native(graph.view_dtype(zero_points.native(), mirtal::DType::Uint32)?)?;
    let arrays = stream.kernels().awq_repack(
        stream,
        [&weight, &zero_points, &scales],
        input,
        output,
        format.group_size,
    )?;
    Ok(QuantizedLinear::from_quantized(arrays, i32::try_from(format.group_size)?, 4))
}

pub(super) fn gptq_linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<QuantizedLinear> {
    let TensorStorage::Gptq { format, scales, zero_points, .. } = &binding.storage else {
        return Err(invalid(binding, "binding is not GPTQ"));
    };
    if format.bits != GptqBits::Four
        || format.scale_dtype != GptqScaleDType::F16
        || !format.symmetric
        || format.activation_order
    {
        return Err(invalid(binding, "format is not the native Metal GPTQ W4A16 contract"));
    }
    let [output, input] = matrix_shape(binding)?;
    let groups = input / format.group_size;
    let graph = stream.native().graph();
    let weight = tensors.get(&binding.source)?;
    let zero_points = tensors.get(zero_points)?;
    let scales = tensors.get(scales)?;
    require(&weight, Dtype::Int32, &[input / 8, output], binding)?;
    require(&zero_points, Dtype::Int32, &[groups, output / 8], binding)?;
    require(&scales, Dtype::Float16, &[groups, output], binding)?;
    let weight = Array::from_native(graph.view_dtype(weight.native(), mirtal::DType::Uint32)?)?;
    let zero_points =
        Array::from_native(graph.view_dtype(zero_points.native(), mirtal::DType::Uint32)?)?;
    let arrays = stream.kernels().gptq_repack(
        stream,
        [&weight, &zero_points, &scales],
        input,
        output,
        format.group_size,
        format.checkpoint_format == GptqCheckpointFormat::Gptq,
    )?;
    Ok(QuantizedLinear::from_quantized(arrays, i32::try_from(format.group_size)?, 4))
}

fn matrix_shape(binding: &TensorBinding) -> Result<[usize; 2]> {
    let logical = binding
        .logical_shape
        .as_deref()
        .ok_or_else(|| invalid(binding, "logical shape is missing"))?;
    let [output, input] = logical else {
        return Err(invalid(binding, "logical shape is not a matrix"));
    };
    Ok([*output, *input])
}

fn arrays(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<(QuantizedArrays, i32, i32)> {
    let (TensorStorage::PackedInt8 {
        format,
        scales,
        zero_points,
        group_indices,
        ..
    }
    | TensorStorage::PackedInt4 {
        format,
        scales,
        zero_points,
        group_indices,
        ..
    }) = &binding.storage
    else {
        return Err(invalid(binding, "binding is not a packed integer"));
    };
    let (group_size, repeat_scales) = native_format(*format, binding)?;
    if format.scale_dtype != CompressedIntegerScaleDType::BF16
        || zero_points.is_some()
        || group_indices.is_some()
    {
        return Err(invalid(binding, "format is not a native Metal packed-integer contract"));
    }
    let logical = binding
        .logical_shape
        .as_deref()
        .ok_or_else(|| invalid(binding, "logical shape is missing"))?;
    let input = logical
        .last()
        .copied()
        .filter(|input| *input > 0 && input.is_multiple_of(group_size))
        .ok_or_else(|| invalid(binding, "logical input width is not group-aligned"))?;
    let mut expected_weight = logical.to_vec();
    let Some(packed) = expected_weight.last_mut() else {
        return Err(invalid(binding, "logical shape is empty"));
    };
    *packed = input
        .checked_mul(usize::from(format.bits.get()))
        .ok_or_else(|| invalid(binding, "packed width overflow"))?
        / 32;
    let mut expected_scales = logical.to_vec();
    let Some(scale_width) = expected_scales.last_mut() else {
        return Err(invalid(binding, "logical shape is empty"));
    };
    *scale_width = if repeat_scales {
        1
    } else {
        input / group_size
    };

    let weight = tensors.get(&binding.source)?;
    let scales = tensors.get(scales)?;
    require(&weight, Dtype::Int32, &expected_weight, binding)?;
    require(&scales, Dtype::Bfloat16, &expected_scales, binding)?;
    let graph = stream.native().graph();
    let weight = Array::from_native(graph.view_dtype(weight.native(), mirtal::DType::Uint32)?)?;
    let scales = if repeat_scales {
        let axis = i32::try_from(expected_scales.len() - 1)?;
        let repeats = i32::try_from(input / group_size)?;
        Array::from_native(graph.repeat(scales.native(), repeats, axis)?)?
    } else {
        scales
    };
    let offset = f32::from(1_u16 << (format.bits.get() - 1));
    let biases = scales.multiply_scalar(-offset, stream)?;
    let group_size = i32::try_from(group_size)?;
    let bits = i32::from(format.bits.get());
    Ok((
        QuantizedArrays::new(weight, scales, biases, group_size, bits)?,
        group_size,
        bits,
    ))
}

fn native_format(
    format: models::weights::CompressedIntegerQuantization,
    binding: &TensorBinding,
) -> Result<(usize, bool)> {
    if format.is_symmetric_channel_int8() {
        return Ok((NATIVE_INT8_GROUP_SIZE, true));
    }
    if format.is_symmetric_group_int4()
        && let CompressedIntegerScaleStrategy::Group { group_size } = format.scale_strategy
    {
        return Ok((group_size, false));
    }
    Err(invalid(binding, "unsupported packed-integer format"))
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
        Err(invalid(
            binding,
            "physical dtype or shape differs from the packed-integer contract",
        ))
    }
}

fn invalid(binding: &TensorBinding, reason: &str) -> Error {
    Error::InvalidQuantization(format!("{}: {reason}", binding.source))
}
