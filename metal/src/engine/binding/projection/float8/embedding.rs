use models::weights::{Float8ActivationScale, Float8Format, TensorBinding, TensorStorage};

use super::{Float8Linear, Float8Operation, invalid, linear};
use crate::engine::{Array, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(in crate::engine) struct Float8Embedding {
    linear: Float8Linear,
}

pub(in crate::engine) fn embedding(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<Float8Embedding> {
    let TensorStorage::Float8 { format, bias: None, .. } = &binding.storage else {
        return Err(invalid(binding, "direct FP8 embedding cannot have output bias"));
    };
    if format.format != Float8Format::E5M2 || format.activation_scale != Float8ActivationScale::None
    {
        return Err(invalid(binding, "selected-row embedding requires E5M2 BF16 activation"));
    }
    let linear = linear(tensors, binding, stream)?;
    if !matches!(&linear.operation, Float8Operation::Direct(_)) {
        return Err(invalid(binding, "selected-row embedding requires direct FP8 storage"));
    }
    Ok(Float8Embedding { linear })
}

impl Float8Embedding {
    pub(in crate::engine) fn lookup(&self, indices: &Array, stream: &Stream) -> Result<Array> {
        let Float8Operation::Direct(linear) = &self.linear.operation else {
            return Err(Error::InvalidQuantization(
                "FP8 embedding lost its direct storage contract".into(),
            ));
        };
        stream.kernels().direct_fp8_embedding(
            &linear.weight,
            &linear.scales,
            indices,
            linear.embedding_spec(),
            stream,
        )
    }

    pub(in crate::engine) fn project(&self, input: &Array, stream: &Stream) -> Result<Array> {
        self.linear.forward(input, stream)
    }
}
