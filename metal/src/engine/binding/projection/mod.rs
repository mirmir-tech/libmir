use architecture::metal::IndividualBank;
use models::weights::{AffineStorageDType, BindingTransform, TensorBinding, TensorStorage};

use crate::engine::{
    Array, DenseLinear, Error, ModelTensors, QuantizedLinear, Result, Stream, linear::GptqLinear,
};
mod bitsandbytes;
mod dense;
mod embedding;
mod float8;
mod fusion;
mod ggml;
mod gptq;
mod graph;
mod mxfp4;
mod mxfp8;
mod nvfp4;
mod packed_integer;
mod routing;
mod rows;
use bitsandbytes::BitsAndBytes4BitLinear;
pub(in crate::engine) use embedding::BoundEmbedding;
use float8::Float8Linear;
use ggml::GgmlLinear;
pub(in crate::engine) use graph::GraphLinear;
use mxfp4::MxFp4Linear;
pub(in crate::engine) use mxfp4::MxFp4LinearLayout;
use mxfp8::MxFp8Linear;
use nvfp4::NvFp4Linear;

#[derive(Debug)]
pub(in crate::engine) enum BoundLinear {
    Dense(DenseLinear),
    Affine(QuantizedLinear),
    Float8(Float8Linear),
    Gptq(GptqLinear),
    BitsAndBytes4Bit(BitsAndBytes4BitLinear),
    MxFp4(MxFp4Linear),
    MxFp8(MxFp8Linear),
    NvFp4(NvFp4Linear),
    Ggml(GgmlLinear),
}
impl BoundLinear {
    /// An expert bank stacked from per-expert projections: dense, NVFP4 or
    /// compressed-tensors packed integers.
    pub(in crate::engine) fn load_individual_bank(
        tensors: &ModelTensors,
        bindings: &[&TensorBinding],
        stream: &Stream,
    ) -> Result<Self> {
        // The bank admission checks for the layout is the one built here.
        match IndividualBank::of(bindings) {
            Some(IndividualBank::Dense) => {
                dense::individual_bank(tensors, bindings, stream).map(Self::Dense)
            },
            Some(IndividualBank::PackedInteger) => {
                packed_integer::individual_bank(tensors, bindings, stream).map(Self::Affine)
            },
            Some(IndividualBank::NvFp4) => {
                nvfp4::individual_bank(tensors, bindings, stream).map(Self::NvFp4)
            },
            None => Err(Error::InvalidQuantization(format!(
                "per-expert projections from {} stack into no Metal expert bank",
                bindings.first().map_or("an empty bank", |binding| binding.source.as_str())
            ))),
        }
    }

    /// A gathered MXFP4 expert bank; see [`MxFp4Linear::bank`].
    pub(in crate::engine) fn mxfp4_bank(
        blocks: &Array,
        scales: Array,
        bias: Array,
        stream: &Stream,
    ) -> Result<Self> {
        MxFp4Linear::bank(blocks, scales, bias, stream).map(Self::MxFp4)
    }

    pub(in crate::engine) fn load(
        tensors: &ModelTensors,
        binding: &TensorBinding,
        stream: &Stream,
    ) -> Result<Self> {
        match &binding.storage {
            TensorStorage::Dense { bias, .. } => DenseLinear::load_binding_names(
                tensors,
                &binding.source,
                bias.as_deref(),
                None,
                binding.transforms.contains(&BindingTransform::Transpose),
                stream,
            )
            .map(Self::Dense),
            TensorStorage::AffineQuantized {
                scales,
                biases: Some(biases),
                output_bias,
                format,
                ..
            } if native_affine(*format) => QuantizedLinear::load_names(
                tensors,
                &binding.source,
                scales,
                biases,
                output_bias.as_deref(),
                i32::try_from(format.group_size)?,
            )
            .map(Self::Affine),
            TensorStorage::PackedInteger { .. } => {
                packed_integer::linear(tensors, binding, stream).map(Self::Affine)
            },
            TensorStorage::Awq { .. } => {
                packed_integer::awq_linear(tensors, binding, stream).map(Self::Affine)
            },
            TensorStorage::Gptq { format, .. } if format.activation_order => {
                gptq::linear(tensors, binding, stream).map(Self::Gptq)
            },
            TensorStorage::Gptq { .. } => {
                packed_integer::gptq_linear(tensors, binding, stream).map(Self::Affine)
            },
            TensorStorage::BitsAndBytes4Bit { .. } => {
                bitsandbytes::linear(tensors, binding).map(Self::BitsAndBytes4Bit)
            },
            TensorStorage::Float8 { .. } => {
                float8::linear(tensors, binding, stream).map(Self::Float8)
            },
            TensorStorage::BlockQuantized { format, .. } if format.is_mxfp4() => {
                mxfp4::linear(tensors, binding, stream).map(Self::MxFp4)
            },
            TensorStorage::BlockQuantized { format, .. }
                if *format == models::weights::BlockQuantization::MXFP8 =>
            {
                mxfp8::linear(tensors, binding).map(Self::MxFp8)
            },
            TensorStorage::BlockQuantized { format, .. }
                if *format == models::weights::BlockQuantization::NVFP4 =>
            {
                match nvfp4::linear(tensors, binding, stream)? {
                    nvfp4::NvFp4Fallback::Dense(linear) => Ok(Self::Dense(linear)),
                    nvfp4::NvFp4Fallback::Gathered(linear) => Ok(Self::NvFp4(linear)),
                }
            },
            TensorStorage::Ggml { .. } => ggml::linear_affine(tensors, binding, stream)?
                .map_or_else(
                    || ggml::linear(tensors, binding, stream).map(Self::Ggml),
                    |linear| Ok(Self::Affine(linear)),
                ),
            _ => Err(unsupported("linear", binding)),
        }
    }

    pub(in crate::engine) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Dense(linear) => linear.forward(input, stream),
            Self::Affine(linear) => linear.forward(input, stream),
            Self::Float8(linear) => linear.forward(input, stream),
            Self::Gptq(linear) => linear.forward(input, stream),
            Self::BitsAndBytes4Bit(linear) => linear.forward(input, stream),
            Self::MxFp4(linear) => linear.forward(input, stream),
            Self::MxFp8(linear) => linear.forward(input, stream),
            Self::Ggml(linear) => linear.forward(input, stream),
            Self::NvFp4(_) => Err(Error::InvalidQuantization(
                "gathered NVFP4 matrix bank does not support ordinary execution".into(),
            )),
        }
    }

    pub(in crate::engine) fn gather(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        stream: &Stream,
    ) -> Result<Array> {
        match self {
            Self::Dense(linear) => linear.gather(input, indices, stream),
            Self::Affine(linear) => linear.gather(input, indices, sorted, stream),
            Self::Float8(linear) => linear.gather(input, indices, stream),
            Self::MxFp4(linear) => linear.gather(input, indices, sorted, stream),
            Self::MxFp8(linear) => linear.gather(input, indices, stream),
            Self::NvFp4(linear) => linear.gather(input, indices, stream),
            Self::BitsAndBytes4Bit(_) => Err(Error::InvalidQuantization(
                "bitsandbytes matrix does not support gathered execution".into(),
            )),
            Self::Gptq(_) => Err(Error::InvalidQuantization(
                "activation-ordered GPTQ does not support gathered execution".into(),
            )),
            Self::Ggml(linear) => linear.gather(input, indices, sorted, stream),
        }
    }

    pub(in crate::engine) fn gather_native(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        stream: &Stream,
    ) -> Result<Array> {
        match self {
            Self::Affine(linear) => linear.gather_native(input, indices, sorted, stream),
            _ => self.gather(input, indices, sorted, stream),
        }
    }

    pub(in crate::engine) fn has_bias(&self) -> bool {
        match self {
            Self::Dense(linear) => linear.has_bias(),
            Self::Affine(linear) => linear.has_bias(),
            Self::Float8(linear) => linear.has_bias(),
            Self::MxFp4(linear) => linear.has_bias(),
            Self::MxFp8(linear) => linear.has_bias(),
            Self::Ggml(linear) => linear.has_bias(),
            Self::Gptq(_) | Self::NvFp4(_) | Self::BitsAndBytes4Bit(_) => false,
        }
    }

    pub(in crate::engine) const fn as_affine(&self) -> Option<&QuantizedLinear> {
        match self {
            Self::Affine(linear) => Some(linear),
            Self::Dense(_)
            | Self::Float8(_)
            | Self::Gptq(_)
            | Self::BitsAndBytes4Bit(_)
            | Self::MxFp4(_)
            | Self::MxFp8(_)
            | Self::NvFp4(_)
            | Self::Ggml(_) => None,
        }
    }

    pub(in crate::engine) fn tuning_format(&self) -> (i32, i32) {
        match self {
            Self::Gptq(linear) => (i32::try_from(linear.group_size()).unwrap_or_default(), 4),
            Self::BitsAndBytes4Bit(_) => (64, 4),
            Self::MxFp4(_) => (32, 4),
            _ => self.as_affine().map_or((0, 0), |linear| (linear.group_size(), linear.bits())),
        }
    }
}
fn unsupported(kind: &str, binding: &TensorBinding) -> Error {
    Error::InvalidQuantization(format!("unsupported {kind} binding {}", binding.source))
}

fn native_affine(format: models::weights::GroupedAffineQuantization) -> bool {
    format.is_mlx_layout()
        && format.has_additive_bias()
        && format.storage_dtype == AffineStorageDType::U32
}
