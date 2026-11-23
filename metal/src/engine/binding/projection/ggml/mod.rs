//! GGML block-quantized projections and embeddings, decoded in place.

mod affine;
mod gather;

pub(in crate::engine::binding::projection) use affine::linear_affine;
use models::weights::{TensorBinding, TensorStorage};

use crate::engine::{
    Array, Dtype, Error, ModelTensors, Result, Stream,
    kernels::ggml::{GGML_GEMV_MAX_TOKENS, GgmlMatrix},
};

/// A `[rows, columns]` GGML matrix in its file layout, with an optional bias;
/// an expert bank stacks `experts` matrices as `[experts * rows, columns]`.
#[derive(Debug)]
pub(in crate::engine) struct GgmlLinear {
    weight: Array,
    matrix: GgmlMatrix,
    bias: Option<Array>,
    /// F32 output scale: `[1]`, or `[experts]` for a bank.
    scale: Option<Array>,
    experts: Option<usize>,
}

impl GgmlLinear {
    /// The matrix whose row `i` is this one's row `rows[i]`.
    pub(in crate::engine) fn take_rows(&self, rows: &Array, stream: &Stream) -> Result<Self> {
        if self.experts.is_some() {
            return Err(Error::InvalidQuantization("expert banks keep their row order".into()));
        }
        Ok(Self {
            weight: self.weight.take(rows, 0, stream)?,
            matrix: self.matrix,
            bias: self.bias.as_ref().map(|bias| bias.take(rows, 0, stream)).transpose()?,
            scale: self.scale.as_ref().map(Array::snapshot).transpose()?,
            experts: None,
        })
    }

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
        let output = match &self.bias {
            Some(bias) => output.add(bias, stream)?,
            None => output,
        };
        match &self.scale {
            Some(scale) => scaled(&output, scale, stream),
            None => Ok(output),
        }
    }

    /// The whole matrix (every matrix of a bank, stacked) as dense `[rows,
    /// columns]` values, for consumers that read weights directly; its
    /// scale is applied, a bias is refused.
    pub(in crate::engine) fn dense(&self, stream: &Stream) -> Result<Array> {
        if self.bias.is_some() {
            return Err(Error::InvalidQuantization(
                "a biased GGML matrix has no dense form".into(),
            ));
        }
        let dense = stream.kernels().ggml_dequantize(
            stream,
            &self.weight,
            self.matrix,
            mirtal::DType::Bfloat16,
        )?;
        match (&self.scale, self.experts) {
            (None, _) => Ok(dense),
            (Some(scale), experts) => {
                let matrices = i32::try_from(experts.unwrap_or(1))?;
                let rows = i32::try_from(self.matrix.rows)? / matrices;
                let shape = [matrices, rows, i32::try_from(self.matrix.columns)?];
                let scaled = scaled(
                    &dense.reshape(&shape, stream)?,
                    &scale.reshape(&[matrices, 1, 1], stream)?,
                    stream,
                )?;
                scaled.reshape(&[shape[0] * rows, shape[2]], stream)
            },
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
    let TensorStorage::Ggml { format, bias, scale } = &binding.storage else {
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
    if bias.is_some() && scale.is_some() {
        return Err(invalid(binding, "a scaled GGML projection carries no bias"));
    }
    let scale = scale
        .as_deref()
        .map(|name| tensors.get(name)?.reshape(&[i32::try_from(experts.unwrap_or(1))?], stream))
        .transpose()?;
    Ok(GgmlLinear { weight, matrix, bias, scale, experts })
}

/// `output * scale`, computed in single precision, in the output's dtype.
fn scaled(output: &Array, scale: &Array, stream: &Stream) -> Result<Array> {
    output
        .astype(Dtype::Float32, stream)?
        .multiply(scale, stream)?
        .astype_like(output, stream)
}

pub(super) fn embedding(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<GgmlEmbedding> {
    let table = linear(tensors, binding, stream)?;
    if table.bias.is_some() || table.scale.is_some() || table.experts.is_some() {
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
