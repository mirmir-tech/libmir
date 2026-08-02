use models::weights::{
    Float8Format, Float8ParameterDType, Float8Quantization, Float8ScaleGranularity,
    Float8ScaleMode, TensorBinding,
};

use super::invalid;
use crate::engine::{Array, Dtype, Result, Stream};

#[derive(Debug, Clone, Copy)]
pub(super) enum Geometry {
    Tensor,
    OutputChannel,
    BlockGrid {
        output_block: usize,
        input_block: usize,
        input_groups: usize,
    },
}

pub(super) struct Prepared {
    pub(super) array: Array,
    pub(super) geometry: Geometry,
}

pub(super) fn valid(format: Float8Quantization, has_scale: bool) -> bool {
    let identity = format.format == Float8Format::E5M2
        && format.scale_mode == Float8ScaleMode::None
        && format.scale_granularity == Float8ScaleGranularity::None
        && format.scale_dtype.is_none()
        && !has_scale;
    let explicit = matches!(
        format.scale_mode,
        Float8ScaleMode::Multiplier | Float8ScaleMode::InverseMultiplier
    ) && !matches!(format.scale_granularity, Float8ScaleGranularity::None)
        && matches!(
            format.scale_dtype,
            Some(Float8ParameterDType::BF16 | Float8ParameterDType::F32)
        )
        && has_scale;
    identity || explicit
}

pub(super) fn prepare(
    scale: Option<Array>,
    format: Float8Quantization,
    output: usize,
    input: usize,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<Prepared> {
    if format.scale_mode == Float8ScaleMode::None {
        let array = Array::from_native(stream.native().graph().full(
            &mirtal::Shape::new([1])?,
            1.0,
            mirtal::DType::Float32,
        )?)?;
        return Ok(Prepared { array, geometry: Geometry::Tensor });
    }
    let scale = scale.ok_or_else(|| invalid(binding, "weight scale is missing"))?;
    require_dtype(&scale, format, binding)?;
    let (reshape, geometry) = geometry(&scale, format.scale_granularity, output, input, binding)?;
    let scale = match reshape {
        Some(shape) => scale.reshape(&shape, stream)?,
        None => scale,
    };
    let scale = scale.astype(Dtype::Float32, stream)?;
    let array = if format.scale_mode == Float8ScaleMode::InverseMultiplier {
        Array::from_native(stream.native().graph().reciprocal(scale.native())?)?
    } else {
        scale
    };
    Ok(Prepared { array, geometry })
}

fn require_dtype(scale: &Array, format: Float8Quantization, binding: &TensorBinding) -> Result<()> {
    let dtype = match format.scale_dtype {
        Some(Float8ParameterDType::BF16) => Dtype::Bfloat16,
        Some(Float8ParameterDType::F32) => Dtype::Float32,
        None => return Err(invalid(binding, "scale dtype is missing")),
    };
    if scale.dtype()? == dtype {
        Ok(())
    } else {
        Err(invalid(binding, "scale dtype differs from the contract"))
    }
}

fn geometry(
    scale: &Array,
    granularity: Float8ScaleGranularity,
    output: usize,
    input: usize,
    binding: &TensorBinding,
) -> Result<(Option<Vec<i32>>, Geometry)> {
    let output_i32 = i32::try_from(output)?;
    match granularity {
        Float8ScaleGranularity::Tensor if scale.shape()?.is_empty() => {
            Ok((Some(vec![1]), Geometry::Tensor))
        },
        Float8ScaleGranularity::Tensor if scale.shape()? == [1] => Ok((None, Geometry::Tensor)),
        Float8ScaleGranularity::OutputChannel if scale.shape()? == [output_i32] => {
            Ok((None, Geometry::OutputChannel))
        },
        Float8ScaleGranularity::OutputChannel if scale.shape()? == [output_i32, 1] => {
            Ok((Some(vec![output_i32]), Geometry::OutputChannel))
        },
        Float8ScaleGranularity::BlockGrid {
            output_groups,
            input_groups,
            output_block_size,
            input_block_size,
        } => block_grid(
            scale,
            [output, input],
            [output_groups, input_groups],
            [output_block_size, input_block_size],
            binding,
        ),
        _ => Err(invalid(binding, "scale shape differs from the contract")),
    }
}

fn block_grid(
    scale: &Array,
    matrix: [usize; 2],
    groups: [usize; 2],
    declared: [Option<usize>; 2],
    binding: &TensorBinding,
) -> Result<(Option<Vec<i32>>, Geometry)> {
    let [output, input] = matrix;
    let [output_groups, input_groups] = groups;
    let [output_block, input_block] = match declared {
        [Some(rows), Some(columns)]
            if rows > 0
                && columns > 0
                && output_groups == output.div_ceil(rows)
                && input_groups == input.div_ceil(columns) =>
        {
            [rows, columns]
        },
        [None, None]
            if output_groups > 0
                && input_groups > 0
                && output.is_multiple_of(output_groups)
                && input.is_multiple_of(input_groups) =>
        {
            [output / output_groups, input / input_groups]
        },
        _ => return Err(invalid(binding, "block-grid geometry is invalid or ambiguous")),
    };
    let expected = [i32::try_from(output_groups)?, i32::try_from(input_groups)?];
    if scale.shape()? != expected {
        return Err(invalid(binding, "block-grid scale shape differs from the contract"));
    }
    Ok((None, Geometry::BlockGrid { output_block, input_block, input_groups }))
}
