use models::weights::{Float8ActivationScale, Float8Format, TensorBinding, TensorStorage};

use self::activation::DirectFloat8Activation;
use crate::engine::{
    Array, DenseLinear, Dtype, Error, ModelTensors, Result, Stream,
    kernels::{DirectFp8EmbeddingSpec, DirectFp8Spec},
};

mod activation;
mod embedding;
mod scale;
pub(super) use embedding::{Float8Embedding, embedding};

#[derive(Debug)]
pub(in crate::engine) struct Float8Linear {
    operation: Float8Operation,
    has_bias: bool,
}

#[derive(Debug)]
enum Float8Operation {
    Dense(DenseLinear),
    Direct(DirectFloat8Linear),
}

#[derive(Debug)]
struct DirectFloat8Linear {
    activation: DirectFloat8Activation,
    weight: Array,
    scales: Array,
    bias: Array,
    input_features: usize,
    output_features: usize,
    scale_geometry: scale::Geometry,
    format: Float8Format,
}

pub(super) fn linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<Float8Linear> {
    let TensorStorage::Float8 { format, scale, input_scale, bias } = &binding.storage else {
        return Err(invalid(binding, "requires a weight scale"));
    };
    let [output, input] = matrix_shape(binding)?;
    if !binding.transforms.is_empty()
        || !scale::valid(*format, scale.is_some())
        || !activation::valid(*format, input_scale.is_some())
    {
        return Err(invalid(binding, "does not match the Metal direct FP8 contract"));
    }
    let weight = tensors.get(&binding.source)?;
    require(&weight, Dtype::Uint8, &[output, input], binding, "weight")?;
    let checkpoint_scale = scale.as_deref().map(|name| tensors.get(name)).transpose()?;
    let prepared_scale = scale::prepare(checkpoint_scale, *format, output, input, binding, stream)?;
    let scale = prepared_scale.array;
    let graph = stream.native().graph();
    let checkpoint_bias = bias
        .as_deref()
        .map(|name| tensors.get(name))
        .transpose()?
        .map(|bias| require_bias(bias, output, binding))
        .transpose()?;
    let has_bias = checkpoint_bias.is_some();
    let operation = if format.format == Float8Format::E4M3
        && format.activation_scale == Float8ActivationScale::None
        && !matches!(prepared_scale.geometry, scale::Geometry::BlockGrid { .. })
    {
        let decoded = graph.from_fp8(weight.native(), mirtal::DType::Float32)?;
        let scale = match prepared_scale.geometry {
            scale::Geometry::Tensor => scale,
            scale::Geometry::OutputChannel => {
                scale.reshape(&[i32::try_from(output)?, 1], stream)?
            },
            scale::Geometry::BlockGrid { .. } => unreachable!("excluded above"),
        };
        let scaled = graph.multiply(&decoded, scale.native())?;
        let weight = Array::from_native(graph.astype(&scaled, mirtal::DType::Bfloat16)?)?;
        weight.async_eval(stream)?;
        Float8Operation::Dense(DenseLinear::from_binding_weight(
            weight, checkpoint_bias, false, stream,
        )?)
    } else {
        let bias = checkpoint_bias.map_or_else(
            || {
                Array::from_native(graph.full(
                    &mirtal::Shape::new([output])?,
                    0.0,
                    mirtal::DType::Bfloat16,
                )?)
            },
            Ok,
        )?;
        scale.async_eval(stream)?;
        bias.async_eval(stream)?;
        let activation = DirectFloat8Activation::prepare(
            tensors,
            *format,
            input_scale.as_deref(),
            binding,
            stream,
        )?;
        Float8Operation::Direct(DirectFloat8Linear {
            activation,
            weight,
            scales: scale,
            bias,
            input_features: input,
            output_features: output,
            scale_geometry: prepared_scale.geometry,
            format: format.format,
        })
    };
    Ok(Float8Linear { operation, has_bias })
}

impl Float8Linear {
    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        match &self.operation {
            Float8Operation::Dense(dense) => dense.forward(input, stream),
            Float8Operation::Direct(linear) => linear.forward(input, stream),
        }
    }

    pub(super) fn gather(&self, input: &Array, indices: &Array, stream: &Stream) -> Result<Array> {
        match &self.operation {
            Float8Operation::Dense(dense) => dense.gather(input, indices, stream),
            Float8Operation::Direct(_) => Err(Error::InvalidQuantization(
                "quantized direct FP8 does not support gathered execution".into(),
            )),
        }
    }

    pub(super) const fn has_bias(&self) -> bool {
        self.has_bias
    }
}

impl DirectFloat8Linear {
    fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        if self.activation.is_bfloat16() {
            if input.dtype()? != Dtype::Bfloat16 {
                return Err(Error::InvalidModel("direct FP8 activation input is not BF16".into()));
            }
            return stream.kernels().direct_fp8(
                [input, &self.scales, &self.weight, &self.scales, &self.bias],
                self.kernel_spec(0, false),
                stream,
            );
        }
        let encoded = self.activation.encode(input, stream)?;
        stream.kernels().direct_fp8(
            [encoded.input(), encoded.scale(), &self.weight, &self.scales, &self.bias],
            self.kernel_spec(encoded.scale_stride(), true),
            stream,
        )
    }

    fn kernel_spec(&self, activation_stride: usize, activation_fp8: bool) -> DirectFp8Spec {
        let (scale_stride, scale_grid, output_block, input_block, input_groups) = self.scale_spec();
        DirectFp8Spec {
            input_features: self.input_features,
            output_features: self.output_features,
            scale_stride,
            scale_grid,
            output_block,
            input_block,
            input_groups,
            activation_stride,
            activation_fp8,
            weight_e5m2: self.format == Float8Format::E5M2,
        }
    }

    fn embedding_spec(&self) -> DirectFp8EmbeddingSpec {
        let (scale_stride, scale_grid, output_block, input_block, input_groups) = self.scale_spec();
        DirectFp8EmbeddingSpec {
            vocab: self.output_features,
            hidden: self.input_features,
            scale_stride,
            scale_grid,
            output_block,
            input_block,
            input_groups,
            weight_e5m2: self.format == Float8Format::E5M2,
        }
    }

    fn scale_spec(&self) -> (usize, bool, usize, usize, usize) {
        match self.scale_geometry {
            scale::Geometry::Tensor => (0, false, 1, 1, 1),
            scale::Geometry::OutputChannel => (1, false, 1, 1, 1),
            scale::Geometry::BlockGrid { output_block, input_block, input_groups } => {
                (0, true, output_block, input_block, input_groups)
            },
        }
    }
}

fn matrix_shape(binding: &TensorBinding) -> Result<[usize; 2]> {
    let Some([output, input]) = binding.logical_shape.as_deref() else {
        return Err(invalid(binding, "logical shape is not a matrix"));
    };
    Ok([*output, *input])
}

fn require(
    array: &Array,
    dtype: Dtype,
    shape: &[usize],
    binding: &TensorBinding,
    kind: &str,
) -> Result<()> {
    let expected = shape
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if array.dtype()? == dtype && array.shape()? == expected {
        Ok(())
    } else {
        Err(invalid(binding, &format!("{kind} dtype or shape differs from the contract")))
    }
}

fn require_bias(bias: Array, output: usize, binding: &TensorBinding) -> Result<Array> {
    require(&bias, Dtype::Bfloat16, &[output], binding, "bias")?;
    Ok(bias)
}

fn invalid(binding: &TensorBinding, reason: &str) -> Error {
    Error::InvalidQuantization(format!("{}: {reason}", binding.source))
}
