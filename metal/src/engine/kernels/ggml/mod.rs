//! Kernels over GGML block-quantized matrices kept in their file layout.
//!
//! Weights are `[rows, row_bytes]` `Uint8` arrays of whole blocks; the
//! decoders in `kernels/ggml/decode.h` read them in place, so device memory
//! equals the GGUF payload. Few-token products run a GEMV that reads each
//! block once for every token; wider batches dequantize to the activation
//! type and use the dense GEMM.

mod launch;
mod source;

use models::gguf::GgmlType;
pub(super) use source::GgmlKernels;

use super::Kernels;
use crate::engine::{Array, Error, Result, Stream};

/// The widest batch the GGML products serve (one token: the GEMV; more: the
/// simdgroup-matrix kernel); wider batches dequantize for the dense GEMM.
pub(in crate::engine) const GGML_GEMV_MAX_TOKENS: usize = 32;

/// GGML block types with Metal decoders.
pub(in crate::engine) const GGML_METAL_TYPES: [GgmlType; 20] = [
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q2_K,
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::IQ4_NL,
    GgmlType::IQ4_XS,
    GgmlType::MXFP4,
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_XS,
    GgmlType::IQ2_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ3_S,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
];

/// IQ1-IQ3 types decode through grid tables, compiled only into the kernels
/// that read them.
pub(super) const fn uses_grids(ggml_type: GgmlType) -> bool {
    matches!(
        ggml_type,
        GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ2_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::IQ1_S
            | GgmlType::IQ1_M
    )
}

/// One GGML matrix: its block type and logical `[rows, columns]` geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) struct GgmlMatrix {
    pub(in crate::engine) ggml_type: GgmlType,
    pub(in crate::engine) rows: usize,
    pub(in crate::engine) columns: usize,
}

impl GgmlMatrix {
    /// Checks the type has a Metal decoder and `weight` holds whole blocks.
    pub(in crate::engine) fn new(
        ggml_type: GgmlType,
        columns: usize,
        weight: &Array,
    ) -> Result<Self> {
        if !GGML_METAL_TYPES.contains(&ggml_type) {
            return Err(Error::InvalidQuantization(format!(
                "GGML type {ggml_type} has no Metal decoder"
            )));
        }
        let shape = weight.shape()?;
        let row_bytes = usize::try_from(ggml_type.row_bytes(u64::try_from(columns)?)?)?;
        let rows = match *shape.as_slice() {
            [rows, bytes] if usize::try_from(bytes)? == row_bytes => usize::try_from(rows)?,
            _ => {
                return Err(Error::InvalidQuantization(format!(
                    "GGML {ggml_type} matrix {shape:?} is not [rows, {row_bytes}]"
                )));
            },
        };
        if weight.native().dtype()? != mirtal::DType::Uint8
            || rows == 0
            || !columns.is_multiple_of(32)
        {
            return Err(Error::InvalidQuantization("invalid GGML matrix storage".into()));
        }
        Ok(Self { ggml_type, rows, columns })
    }
}

impl Kernels {
    /// `input [.., columns] · weightᵀ` for at most [`GGML_GEMV_MAX_TOKENS`]
    /// rows.
    pub(crate) fn ggml_gemv(
        &self,
        stream: &Stream,
        input: &Array,
        weight: &Array,
        matrix: GgmlMatrix,
    ) -> Result<Array> {
        launch::gemv(
            [self.ggml.gemv.select(matrix.ggml_type), self.ggml.mma.select(matrix.ggml_type)],
            stream,
            input,
            weight,
            matrix,
        )
    }

    /// Routed products over an expert bank of `experts` stacked matrices:
    /// assignment `a` multiplies input row `a / (assignments / input rows)`
    /// by matrix `indices[a]`, giving `[assignments, rows per expert]`.
    pub(crate) fn ggml_gathered(
        &self,
        stream: &Stream,
        inputs: [&Array; 3],
        matrix: GgmlMatrix,
        experts: usize,
    ) -> Result<Array> {
        launch::gathered(
            self.ggml.gathered.select(matrix.ggml_type),
            stream,
            inputs,
            matrix,
            experts,
        )
    }

    /// Routed products for expert-sorted assignments, one input row each:
    /// `[assignments, rows per expert]` (prefill).
    pub(crate) fn ggml_grouped(
        &self,
        stream: &Stream,
        inputs: [&Array; 3],
        matrix: GgmlMatrix,
        experts: usize,
    ) -> Result<Array> {
        launch::grouped(self.ggml.grouped.select(matrix.ggml_type), stream, inputs, matrix, experts)
    }

    /// The matrix as dense `[rows, columns]` values of `dtype`.
    pub(crate) fn ggml_dequantize(
        &self,
        stream: &Stream,
        weight: &Array,
        matrix: GgmlMatrix,
        dtype: mirtal::DType,
    ) -> Result<Array> {
        launch::dequantize(
            self.ggml.dequantize.select(matrix.ggml_type),
            stream,
            weight,
            matrix,
            dtype,
        )
    }

    /// GGML MXFP4 rows `[rows, blocks * 17]` as OCP `[rows, blocks, 16]`
    /// nibbles and `[rows, blocks]` E8M0 scales.
    pub(crate) fn ggml_mxfp4_repack(&self, stream: &Stream, weight: &Array) -> Result<[Array; 2]> {
        launch::mxfp4_repack(&self.ggml.mxfp4_repack, stream, weight)
    }

    /// Rows `indices` of an embedding table as `indices.shape + [columns]`.
    pub(crate) fn ggml_embedding(
        &self,
        stream: &Stream,
        weight: &Array,
        indices: &Array,
        matrix: GgmlMatrix,
        dtype: mirtal::DType,
    ) -> Result<Array> {
        launch::embedding(
            self.ggml.embedding.select(matrix.ggml_type),
            stream,
            weight,
            indices,
            matrix,
            dtype,
        )
    }
}

#[cfg(test)]
mod tests;
