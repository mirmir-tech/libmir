use models::weights::{
    Float8ActivationScale, Float8Format, Float8ParameterDType, Float8Quantization,
    Float8ScaleGranularity, Float8ScaleMode,
};

use crate::{
    Error, Result,
    kernels::{DirectFp8Activation, DirectFp8Format, DirectFp8Scale},
};

pub(super) fn execution_contract(
    name: &str,
    format: Float8Quantization,
    input: usize,
    output: usize,
) -> Result<(DirectFp8Format, DirectFp8Scale, bool, DirectFp8Activation)> {
    let value_format = match format.format {
        Float8Format::E4M3 => DirectFp8Format::E4M3,
        Float8Format::E5M2 => DirectFp8Format::E5M2,
    };
    let identity = format.scale_mode == Float8ScaleMode::None
        && format.scale_granularity == Float8ScaleGranularity::None
        && format.scale_dtype.is_none();
    if identity {
        return activation_contract(name, value_format, DirectFp8Scale::Tensor, false, format);
    }
    if format.scale_mode == Float8ScaleMode::None
        || format.scale_granularity == Float8ScaleGranularity::None
        || !matches!(
            format.scale_dtype,
            Some(Float8ParameterDType::BF16 | Float8ParameterDType::F32)
        )
    {
        return Err(unsupported(name, "has an incomplete scale contract"));
    }
    let inverse = format.scale_mode == Float8ScaleMode::InverseMultiplier;
    let scale = match format.scale_granularity {
        Float8ScaleGranularity::Tensor => DirectFp8Scale::Tensor,
        Float8ScaleGranularity::OutputChannel => DirectFp8Scale::OutputChannel,
        Float8ScaleGranularity::BlockGrid {
            output_groups,
            input_groups,
            output_block_size,
            input_block_size,
        } => block_scale(
            name,
            input,
            output,
            output_groups,
            input_groups,
            output_block_size,
            input_block_size,
        )?,
        Float8ScaleGranularity::None => unreachable!("validated above"),
    };
    activation_contract(name, value_format, scale, inverse, format)
}

#[allow(clippy::too_many_arguments)]
fn block_scale(
    name: &str,
    input: usize,
    output: usize,
    output_groups: usize,
    input_groups: usize,
    declared_output_block: Option<usize>,
    declared_input_block: Option<usize>,
) -> Result<DirectFp8Scale> {
    let (output_block_size, input_block_size) = match (declared_output_block, declared_input_block)
    {
        (Some(rows), Some(columns))
            if rows > 0
                && columns.is_multiple_of(4)
                && output_groups == output.div_ceil(rows)
                && input_groups == input.div_ceil(columns) =>
        {
            (rows, columns)
        },
        (None, None)
            if output_groups > 0
                && input_groups > 0
                && output.is_multiple_of(output_groups)
                && input.is_multiple_of(input_groups)
                && (input / input_groups).is_multiple_of(4) =>
        {
            (output / output_groups, input / input_groups)
        },
        _ => return Err(unsupported(name, "has invalid or ambiguous block-grid geometry")),
    };
    Ok(DirectFp8Scale::BlockGrid {
        output_groups,
        input_groups,
        output_block_size,
        input_block_size,
    })
}

fn activation_contract(
    name: &str,
    value_format: DirectFp8Format,
    scale: DirectFp8Scale,
    inverse: bool,
    format: Float8Quantization,
) -> Result<(DirectFp8Format, DirectFp8Scale, bool, DirectFp8Activation)> {
    let activation = match format.activation_scale {
        Float8ActivationScale::None => DirectFp8Activation::Bf16,
        Float8ActivationScale::DynamicToken if value_format == DirectFp8Format::E4M3 => {
            DirectFp8Activation::DynamicE4M3Token
        },
        Float8ActivationScale::DynamicToken => {
            return Err(unsupported(name, "does not support dynamic E5M2 activations"));
        },
        Float8ActivationScale::StaticTensor
            if value_format == DirectFp8Format::E4M3
                && matches!(
                    format.input_scale_dtype,
                    Some(Float8ParameterDType::BF16 | Float8ParameterDType::F32)
                )
                && format.input_scale_dtype == format.scale_dtype =>
        {
            DirectFp8Activation::StaticE4M3Tensor
        },
        Float8ActivationScale::StaticTensor => {
            return Err(unsupported(
                name,
                "requires a static E4M3 activation scale matching the weight scale dtype",
            ));
        },
    };
    Ok((value_format, scale, inverse, activation))
}

pub(super) fn unsupported(name: &str, requirement: &str) -> Error {
    Error::UnsupportedDecoderLayer(format!("direct FP8 projection {name} {requirement}"))
}
