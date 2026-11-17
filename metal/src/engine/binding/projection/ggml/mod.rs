//! GGML block-quantized projections and embeddings, decoded in place.

mod gather;

use models::weights::{TensorBinding, TensorStorage};

use crate::engine::{
    Array, Error, ModelTensors, Result, Stream,
    kernels::ggml::{GGML_GEMV_MAX_TOKENS, GgmlMatrix},
};

/// A `[rows, columns]` GGML matrix in its file layout, with an optional bias;
/// an expert bank stacks `experts` matrices as `[experts * rows, columns]`.
#[derive(Debug)]
pub(in crate::engine) struct GgmlLinear {
    weight: Array,
    matrix: GgmlMatrix,
    bias: Option<Array>,
    experts: Option<usize>,
}

impl GgmlLinear {
    pub(in crate::engine) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        if self.experts.is_some() {
            return Err(Error::InvalidQuantization(
                "GGML expert bank requires gathered execution".into(),
            ));
        }
        let output = if tokens(input)? <= GGML_GEMV_MAX_TOKENS {
            stream.kernels().ggml_gemv(stream, input, &self.weight, self.matrix)?
        } else {
            let dense = stream.kernels().ggml_dequantize(
                stream,
                &self.weight,
                self.matrix,
                input.native().dtype()?,
            )?;
            input.matmul(&dense.transpose(&[1, 0], stream)?, stream)?
        };
        match &self.bias {
            Some(bias) => output.add(bias, stream),
            None => Ok(output),
        }
    }

    pub(in crate::engine) const fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

/// A GGML embedding table: lookups decode only the selected rows, and the
/// tied output projection runs the GGML product over the same blocks.
#[derive(Debug)]
pub(in crate::engine) struct GgmlEmbedding {
    table: GgmlLinear,
}

impl GgmlEmbedding {
    pub(in crate::engine) fn lookup(&self, indices: &Array, stream: &Stream) -> Result<Array> {
        stream.kernels().ggml_embedding(
            stream,
            &self.table.weight,
            indices,
            self.table.matrix,
            mirtal::DType::Bfloat16,
        )
    }

    pub(in crate::engine) fn project(&self, input: &Array, stream: &Stream) -> Result<Array> {
        self.table.forward(input, stream)
    }
}

pub(super) fn linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<GgmlLinear> {
    let TensorStorage::Ggml { format, bias } = &binding.storage else {
        return Err(invalid(binding, "binding is not GGML"));
    };
    let shape = binding.logical_shape.as_deref().unwrap_or(&binding.shape);
    let (experts, rows, columns) = match *shape {
        [rows, columns] => (None, rows, columns),
        [experts, rows, columns] => (Some(experts), experts * rows, columns),
        _ => return Err(invalid(binding, "GGML projection is not a matrix or a bank")),
    };
    let mut weight = tensors.get(&binding.source)?;
    if experts.is_some() {
        let bytes = *weight.shape()?.last().ok_or(Error::ShapeOverflow)?;
        weight = weight.reshape(&[i32::try_from(rows)?, bytes], stream)?;
    }
    let matrix = GgmlMatrix::new(format.ggml_type, columns, &weight)?;
    if matrix.rows != rows {
        return Err(invalid(binding, "GGML row count differs from the binding"));
    }
    let bias = bias.as_deref().map(|name| tensors.get(name)).transpose()?;
    Ok(GgmlLinear { weight, matrix, bias, experts })
}

pub(super) fn embedding(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<GgmlEmbedding> {
    let table = linear(tensors, binding, stream)?;
    if table.bias.is_some() || table.experts.is_some() {
        return Err(invalid(binding, "GGML embedding is not a plain matrix"));
    }
    Ok(GgmlEmbedding { table })
}

fn tokens(input: &Array) -> Result<usize> {
    let shape = input.shape()?;
    shape[..shape.len().saturating_sub(1)].iter().try_fold(1_usize, |total, value| {
        total.checked_mul(usize::try_from(*value)?).ok_or(Error::ShapeOverflow)
    })
}

fn invalid(binding: &TensorBinding, reason: &str) -> Error {
    Error::InvalidQuantization(format!("{}: {reason}", binding.source))
}

#[cfg(test)]
mod tests;
