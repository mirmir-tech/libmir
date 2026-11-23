//! BF16 projections and embedding rows over GGML blocks read in place from
//! GGUF files (`kernels/ggml/`).

mod dequantize;
mod embedding;
mod gathered;
mod gemv;
mod linear;
mod mmq;
mod repack;
mod scale;
#[cfg(test)]
mod tests;

pub use dequantize::GgmlDequantize;
pub use embedding::{GgmlEmbedding, GgmlEmbeddingOperands};
pub use gathered::{GgmlExpertSpec, GgmlGatheredLinear, GgmlSelections};
pub use linear::{GgmlActivations, GgmlLinear};
pub use mmq::{GgmlGroupedMmq, GgmlMmqGroups, GgmlMmqInput, GgmlMmqPlacement, GgmlRouteCombine};
use models::gguf::GgmlType;
pub use repack::{GgmlMxFp4Repack, GgmlRowPlacement};
pub use scale::{GgmlOutputScaling, GgmlScaledRows};

/// Where one projection's rows sit in each expert's matrix of a bank.
///
/// The matrix holds `expert_rows` rows and the projection's start at
/// `first_row`; a fused gate/up bank holds the gate rows, then the up rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgmlBankRows {
    pub expert_rows: usize,
    pub first_row: usize,
}

impl GgmlBankRows {
    /// A bank whose experts' matrices are exactly the projection.
    #[must_use]
    pub const fn whole(rows: usize) -> Self {
        Self { expert_rows: rows, first_row: 0 }
    }

    /// Whether `rows` projection rows fit each expert's matrix.
    fn check(self, rows: usize) -> Result<()> {
        if self.first_row.checked_add(rows).is_some_and(|end| end <= self.expert_rows) {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan("GGML projection rows exceed the expert matrix"))
        }
    }
}

use super::geometry::product;
use crate::{Error, Result};

/// Batches up to this many tokens use the GEMV; larger ones dequantize the
/// matrix to BF16 for the dense GEMM.
pub const GGML_GEMV_MAX_TOKENS: usize = 16;

/// A GGML type with CUDA kernels, as admission declares them
/// (`architecture::ggml::CUDA`).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GgmlKernelType(GgmlType);

impl GgmlKernelType {
    /// Every GGML type with CUDA kernels.
    pub fn all() -> impl Iterator<Item = Self> {
        architecture::ggml::CUDA.iter().map(|&ggml_type| Self(ggml_type))
    }

    /// The kernel family of a GGML type; a type without kernels is refused
    /// by name.
    pub fn new(ggml_type: GgmlType) -> Result<Self> {
        if architecture::ggml::CUDA.contains(&ggml_type) {
            Ok(Self(ggml_type))
        } else {
            Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA has no kernels for GGML {ggml_type} blocks yet"
            )))
        }
    }

    #[must_use]
    pub const fn ggml_type(self) -> GgmlType {
        self.0
    }

    /// Payload bytes of `rows` rows of `columns` values.
    pub fn bytes(self, rows: usize, columns: usize) -> Result<usize> {
        let ggml_type = self.ggml_type();
        let row = ggml_type.row_bytes(u64::try_from(columns)?)?;
        product(rows, usize::try_from(row)?)
    }
}

/// Fixed BF16-by-GGML projection geometry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GgmlSpec {
    pub kind: GgmlKernelType,
    pub tokens: usize,
    pub input_features: usize,
    pub output_features: usize,
}

impl GgmlSpec {
    pub fn new(
        kind: GgmlKernelType,
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        let block = usize::try_from(kind.ggml_type().block_elements())?;
        if tokens == 0
            || output_features == 0
            || input_features == 0
            || !input_features.is_multiple_of(block)
        {
            return Err(Error::InvalidDecoderKernel("invalid GGML projection geometry"));
        }
        Ok(Self {
            kind,
            tokens,
            input_features,
            output_features,
        })
    }

    pub fn input_elements(self) -> Result<usize> {
        product(self.tokens, self.input_features)
    }

    pub fn weight_bytes(self) -> Result<usize> {
        self.kind.bytes(self.output_features, self.input_features)
    }

    pub fn output_elements(self) -> Result<usize> {
        product(self.tokens, self.output_features)
    }
}
