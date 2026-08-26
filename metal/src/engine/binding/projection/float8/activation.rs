use models::weights::{
    Float8ActivationScale, Float8Format, Float8ParameterDType, Float8Quantization, TensorBinding,
};

use super::{invalid, require};
use crate::engine::{Array, Dtype, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) enum DirectFloat8Activation {
    Bfloat16,
    Dynamic,
    Static(Array),
}

pub(super) struct EncodedActivation<'a> {
    input: Array,
    scale: Scale<'a>,
}

enum Scale<'a> {
    Dynamic(Array),
    Static(&'a Array),
}

pub(super) fn valid(format: Float8Quantization, has_input_scale: bool) -> bool {
    match (format.format, format.activation_scale) {
        (Float8Format::E4M3, Float8ActivationScale::None | Float8ActivationScale::DynamicToken) => {
            !has_input_scale
        },
        (Float8Format::E5M2, Float8ActivationScale::None) => !has_input_scale,
        (Float8Format::E4M3, Float8ActivationScale::StaticTensor) => {
            has_input_scale
                && matches!(
                    format.input_scale_dtype,
                    Some(Float8ParameterDType::BF16 | Float8ParameterDType::F32)
                )
                && format.input_scale_dtype == format.scale_dtype
        },
        _ => false,
    }
}

impl DirectFloat8Activation {
    pub(super) fn prepare(
        tensors: &ModelTensors,
        format: Float8Quantization,
        input_scale: Option<&str>,
        binding: &TensorBinding,
        stream: &Stream,
    ) -> Result<Self> {
        match format.activation_scale {
            Float8ActivationScale::DynamicToken => Ok(Self::Dynamic),
            Float8ActivationScale::StaticTensor => {
                let name = input_scale.ok_or_else(|| invalid(binding, "input scale is missing"))?;
                let scale = tensors.get(name)?;
                let dtype = match format.input_scale_dtype {
                    Some(Float8ParameterDType::BF16) => Dtype::Bfloat16,
                    Some(Float8ParameterDType::F32) => Dtype::Float32,
                    None => return Err(invalid(binding, "input scale dtype is missing")),
                };
                if !scale.shape()?.is_empty() && scale.shape()? != [1] {
                    return Err(invalid(binding, "input scale is not scalar"));
                }
                require(&scale, dtype, &[], binding, "input scale")
                    .or_else(|_| require(&scale, dtype, &[1], binding, "input scale"))?;
                let scale = scale.astype(Dtype::Float32, stream)?.reshape(&[1, 1], stream)?;
                scale.async_eval(stream)?;
                Ok(Self::Static(scale))
            },
            Float8ActivationScale::None => Ok(Self::Bfloat16),
        }
    }

    pub(super) fn encode<'a>(
        &'a self,
        input: &Array,
        stream: &Stream,
    ) -> Result<EncodedActivation<'a>> {
        match self {
            Self::Bfloat16 => Err(crate::engine::Error::InvalidQuantization(
                "BF16 direct FP8 activation does not require encoding".into(),
            )),
            Self::Dynamic => {
                let (input, scale) = dynamic(input, stream)?;
                Ok(EncodedActivation { input, scale: Scale::Dynamic(scale) })
            },
            Self::Static(scale) => Ok(EncodedActivation {
                input: encode(input, scale, stream)?,
                scale: Scale::Static(scale),
            }),
        }
    }

    pub(super) const fn is_bfloat16(&self) -> bool {
        matches!(self, Self::Bfloat16)
    }
}

impl EncodedActivation<'_> {
    pub(super) const fn input(&self) -> &Array {
        &self.input
    }

    pub(super) const fn scale(&self) -> &Array {
        match &self.scale {
            Scale::Dynamic(scale) => scale,
            Scale::Static(scale) => scale,
        }
    }

    pub(super) const fn scale_stride(&self) -> usize {
        match self.scale {
            Scale::Dynamic(_) => 1,
            Scale::Static(_) => 0,
        }
    }
}

fn dynamic(input: &Array, stream: &Stream) -> Result<(Array, Array)> {
    const FP8_MAX: f32 = 448.0;
    const MINIMUM_SCALE: f32 = 1.0 / (FP8_MAX * 512.0);

    let graph = stream.native().graph();
    let input_f32 = graph.astype(input.native(), mirtal::DType::Float32)?;
    let absolute = graph.maximum(&input_f32, &graph.negative(&input_f32)?)?;
    let maximum = graph.reduce_max(&absolute, -1, true)?;
    let scale = graph.multiply_scalar(&maximum, 1.0 / FP8_MAX)?;
    let minimum = graph.full(&mirtal::Shape::new([])?, MINIMUM_SCALE, mirtal::DType::Float32)?;
    let scale = graph.maximum(&scale, &minimum)?;
    Ok((encode_native(&input_f32, &scale, stream)?, Array::from_native(scale)?))
}

fn encode(input: &Array, scale: &Array, stream: &Stream) -> Result<Array> {
    let input = stream.native().graph().astype(input.native(), mirtal::DType::Float32)?;
    encode_native(&input, scale.native(), stream)
}

fn encode_native(input: &mirtal::Array, scale: &mirtal::Array, stream: &Stream) -> Result<Array> {
    let graph = stream.native().graph();
    let normalized = graph.divide(input, scale)?;
    Array::from_native(graph.to_fp8(&normalized)?)
}
