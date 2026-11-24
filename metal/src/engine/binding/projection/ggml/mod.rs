//! GGML block-quantized projections and embeddings, decoded in place.

mod affine;
mod gather;

pub(in crate::engine::binding::projection) use affine::linear_affine;
use models::weights::{GgmlCompanion as StoredCompanion, TensorBinding, TensorStorage};

use crate::engine::{
    Array, Dtype, Error, ModelTensors, Result, Stream,
    kernels::ggml::{GGML_GEMV_MAX_TOKENS, GgmlMatrix},
};

/// A `[rows, columns]` GGML matrix in its file layout, or an expert bank of
/// `experts` matrices stacked as `[experts * rows, columns]`.
#[derive(Debug)]
pub(in crate::engine) struct GgmlLinear {
    weight: Array,
    matrix: GgmlMatrix,
    layout: GgmlLayout,
}

/// What a GGML projection is and what it adds to its products.
#[derive(Debug)]
enum GgmlLayout {
    /// One matrix, with at most one companion.
    Matrix(Option<Companion>),
    /// Stacked expert matrices with an optional `[experts]` F32 output scale.
    Bank { experts: usize, scale: Option<Array> },
}

#[derive(Debug)]
enum Companion {
    Bias(Array),
    /// `[1]` F32 output scale.
    Scale(Array),
}

impl GgmlLinear {
    /// The matrix whose row `i` is this one's row `rows[i]`.
    pub(in crate::engine) fn take_rows(&self, rows: &Array, stream: &Stream) -> Result<Self> {
        let GgmlLayout::Matrix(companion) = &self.layout else {
            return Err(Error::InvalidQuantization("expert banks keep their row order".into()));
        };
        let companion = match companion {
            Some(Companion::Bias(bias)) => Some(Companion::Bias(bias.take(rows, 0, stream)?)),
            Some(Companion::Scale(scale)) => Some(Companion::Scale(scale.snapshot()?)),
            None => None,
        };
        Ok(Self {
            weight: self.weight.take(rows, 0, stream)?,
            matrix: self.matrix,
            layout: GgmlLayout::Matrix(companion),
        })
    }

    pub(in crate::engine) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let GgmlLayout::Matrix(companion) = &self.layout else {
            return Err(Error::InvalidQuantization(
                "GGML expert bank requires gathered execution".into(),
            ));
        };
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
        match companion {
            Some(Companion::Bias(bias)) => output.add(bias, stream),
            Some(Companion::Scale(scale)) => scaled(&output, scale, stream),
            None => Ok(output),
        }
    }

    /// The whole matrix (every matrix of a bank, stacked) as dense `[rows,
    /// columns]` values, for consumers that read weights directly; its
    /// scale is applied, a bias is refused.
    pub(in crate::engine) fn dense(&self, stream: &Stream) -> Result<Array> {
        let (matrices, scale) = match &self.layout {
            GgmlLayout::Matrix(Some(Companion::Bias(_))) => {
                return Err(Error::InvalidQuantization(
                    "a biased GGML matrix has no dense form".into(),
                ));
            },
            GgmlLayout::Matrix(Some(Companion::Scale(scale))) => (1, Some(scale)),
            GgmlLayout::Matrix(None) => (1, None),
            GgmlLayout::Bank { experts, scale } => (*experts, scale.as_ref()),
        };
        let dense = stream.kernels().ggml_dequantize(
            stream,
            &self.weight,
            self.matrix,
            mirtal::DType::Bfloat16,
        )?;
        match scale {
            None => Ok(dense),
            Some(scale) => {
                let matrices = i32::try_from(matrices)?;
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
        matches!(self.layout, GgmlLayout::Matrix(Some(Companion::Bias(_))))
    }

    /// The matrix's bias, for conversions that carry it over.
    pub(super) fn into_bias(self) -> Option<Array> {
        match self.layout {
            GgmlLayout::Matrix(Some(Companion::Bias(bias))) => Some(bias),
            GgmlLayout::Matrix(_) | GgmlLayout::Bank { .. } => None,
        }
    }

    pub(super) const fn is_bank(&self) -> bool {
        matches!(self.layout, GgmlLayout::Bank { .. })
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
    let TensorStorage::Ggml { format, companion } = &binding.storage else {
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
    let output_scale = |name: &str, count: usize| -> Result<Array> {
        tensors.get(name)?.reshape(&[i32::try_from(count)?], stream)
    };
    let layout = match (experts, companion) {
        (Some(_), StoredCompanion::Bias(_)) => {
            return Err(invalid(binding, "GGML expert banks carry no bias"));
        },
        (Some(experts), StoredCompanion::Scale(scale)) => GgmlLayout::Bank {
            experts,
            scale: Some(output_scale(scale, experts)?),
        },
        (Some(experts), StoredCompanion::Plain) => GgmlLayout::Bank { experts, scale: None },
        (None, StoredCompanion::Bias(bias)) => {
            GgmlLayout::Matrix(Some(Companion::Bias(tensors.get(bias)?)))
        },
        (None, StoredCompanion::Scale(scale)) => {
            GgmlLayout::Matrix(Some(Companion::Scale(output_scale(scale, 1)?)))
        },
        (None, StoredCompanion::Plain) => GgmlLayout::Matrix(None),
    };
    Ok(GgmlLinear { weight, matrix, layout })
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
    if !matches!(table.layout, GgmlLayout::Matrix(None)) {
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
