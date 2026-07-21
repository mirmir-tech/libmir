use models::weights::{TensorBinding, TensorStorage};

use crate::engine::{
    Array, DenseEmbedding, DenseLinear, ModelTensors, QuantizedEmbedding, QuantizedLinear, Result,
    Stream,
};

#[derive(Debug)]
pub(super) enum BoundLinear {
    Dense(DenseLinear),
    Quantized(QuantizedLinear),
}

impl BoundLinear {
    pub(super) fn load_binding(
        tensors: &ModelTensors,
        binding: &TensorBinding,
        stream: &Stream,
    ) -> Result<Self> {
        match &binding.storage {
            TensorStorage::Dense { bias, .. } => {
                DenseLinear::load_names(tensors, &binding.source, bias.as_deref(), None, stream)
                    .map(Self::Dense)
            },
            TensorStorage::AffineQuantized {
                scales,
                biases: Some(biases),
                output_bias,
                group_size: Some(group_size),
                ..
            } => QuantizedLinear::load_names(
                tensors,
                &binding.source,
                scales,
                biases,
                output_bias.as_deref(),
                i32::try_from(*group_size)?,
            )
            .map(Self::Quantized),
            _ => Err(crate::engine::Error::InvalidQuantization(format!(
                "unsupported clamped-routed linear binding {}",
                binding.source
            ))),
        }
    }

    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Dense(linear) => linear.forward(input, stream),
            Self::Quantized(linear) => linear.forward(input, stream),
        }
    }
}

#[derive(Debug)]
pub(super) enum BoundEmbedding {
    Dense(DenseEmbedding),
    Quantized(QuantizedEmbedding),
}

impl BoundEmbedding {
    pub(super) fn load_binding(tensors: &ModelTensors, binding: &TensorBinding) -> Result<Self> {
        match &binding.storage {
            TensorStorage::Dense { .. } => {
                DenseEmbedding::load_name(tensors, &binding.source).map(Self::Dense)
            },
            TensorStorage::AffineQuantized {
                scales,
                biases: Some(biases),
                group_size: Some(group_size),
                ..
            } => QuantizedEmbedding::load_names(
                tensors,
                &binding.source,
                scales,
                biases,
                i32::try_from(*group_size)?,
            )
            .map(Self::Quantized),
            _ => Err(crate::engine::Error::InvalidQuantization(format!(
                "unsupported clamped-routed embedding binding {}",
                binding.source
            ))),
        }
    }

    pub(super) fn lookup(&self, indices: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Dense(embedding) => embedding.lookup(indices, stream),
            Self::Quantized(embedding) => embedding.lookup(indices, stream),
        }
    }
}
