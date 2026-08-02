use models::weights::{TensorBinding, TensorStorage};

use super::{
    float8::Float8Embedding, mxfp4::MxFp4Embedding, mxfp8::MxFp8Embedding, native_affine,
    unsupported,
};
use crate::engine::{
    Array, DenseEmbedding, DenseLinear, ModelTensors, QuantizedEmbedding, Result, Stream,
};

#[derive(Debug)]
pub(in crate::engine) enum BoundEmbedding {
    Dense {
        lookup: DenseEmbedding,
        projection: DenseLinear,
    },
    Affine(QuantizedEmbedding),
    Float8(Float8Embedding),
    MxFp4(MxFp4Embedding),
    MxFp8(MxFp8Embedding),
}

impl BoundEmbedding {
    pub(in crate::engine) fn load(
        tensors: &ModelTensors,
        binding: &TensorBinding,
        stream: &Stream,
    ) -> Result<Self> {
        match &binding.storage {
            TensorStorage::Dense { .. } => Ok(Self::Dense {
                lookup: DenseEmbedding::load_name(tensors, &binding.source)?,
                projection: DenseLinear::load_names(tensors, &binding.source, None, None, stream)?,
            }),
            TensorStorage::AffineQuantized { scales, biases: Some(biases), format, .. }
                if native_affine(*format) =>
            {
                QuantizedEmbedding::load_names(
                    tensors,
                    &binding.source,
                    scales,
                    biases,
                    i32::try_from(format.group_size)?,
                )
                .map(Self::Affine)
            },
            TensorStorage::PackedInt8 { .. } | TensorStorage::PackedInt4 { .. } => {
                super::packed_integer::embedding(tensors, binding, stream).map(Self::Affine)
            },
            TensorStorage::Float8 { .. } => {
                super::float8::embedding(tensors, binding, stream).map(Self::Float8)
            },
            TensorStorage::BlockQuantized { format, .. } if format.is_mxfp4() => {
                super::mxfp4::embedding(tensors, binding, stream).map(Self::MxFp4)
            },
            TensorStorage::BlockQuantized { format, .. }
                if *format == models::weights::BlockQuantization::MXFP8 =>
            {
                super::mxfp8::embedding(tensors, binding).map(Self::MxFp8)
            },
            _ => Err(unsupported("embedding", binding)),
        }
    }

    pub(in crate::engine) fn lookup(&self, indices: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Dense { lookup, .. } => lookup.lookup(indices, stream),
            Self::Affine(embedding) => embedding.lookup(indices, stream),
            Self::Float8(embedding) => embedding.lookup(indices, stream),
            Self::MxFp4(embedding) => embedding.lookup(indices, stream),
            Self::MxFp8(embedding) => embedding.lookup(indices, stream),
        }
    }

    pub(in crate::engine) fn project(&self, input: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Dense { projection, .. } => projection.forward(input, stream),
            Self::Affine(embedding) => embedding.project(input, stream),
            Self::Float8(embedding) => embedding.project(input, stream),
            Self::MxFp4(embedding) => embedding.project(input, stream),
            Self::MxFp8(embedding) => embedding.project(input, stream),
        }
    }
}
