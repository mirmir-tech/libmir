//! BF16 projections and embedding rows over GGML blocks read in place from
//! GGUF files (`kernels/ggml_bf16.cu`).

mod dequantize;
mod embedding;
mod gathered;
mod gemv;
mod linear;
mod mmq;
mod repack;
#[cfg(test)]
mod tests;

pub use dequantize::GgmlDequantize;
pub use embedding::{GgmlEmbedding, GgmlEmbeddingOperands};
pub use gathered::{GgmlExpertSpec, GgmlGatheredLinear, GgmlSelections};
pub use linear::{GgmlActivations, GgmlLinear};
pub use mmq::{GgmlGroupedMmq, GgmlMmqGroups, GgmlMmqInput, GgmlMmqPlacement, GgmlRouteCombine};
use models::gguf::GgmlType;
pub use repack::{GgmlMxFp4Repack, GgmlRowPlacement};

use super::geometry::product;
use crate::{Error, Result};

/// Batches up to this many tokens use the GEMV; larger ones dequantize the
/// matrix to BF16 for the dense GEMM.
pub const GGML_GEMV_MAX_TOKENS: usize = 16;

/// A GGML type with CUDA kernels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GgmlKernelType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ4NL,
    IQ4XS,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ1S,
    IQ1M,
    MXFP4,
}

impl GgmlKernelType {
    /// Every GGML type with CUDA kernels.
    pub const ALL: [Self; 20] = [
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::IQ4NL,
        Self::IQ4XS,
        Self::IQ2XXS,
        Self::IQ2XS,
        Self::IQ2S,
        Self::IQ3XXS,
        Self::IQ3S,
        Self::IQ1S,
        Self::IQ1M,
        Self::MXFP4,
    ];

    /// The kernel family of a GGML type; `MissingCapability`-style refusal for
    /// types without kernels names the type.
    pub fn new(ggml_type: GgmlType) -> Result<Self> {
        match ggml_type {
            GgmlType::Q4_0 => Ok(Self::Q4_0),
            GgmlType::Q4_1 => Ok(Self::Q4_1),
            GgmlType::Q5_0 => Ok(Self::Q5_0),
            GgmlType::Q5_1 => Ok(Self::Q5_1),
            GgmlType::Q8_0 => Ok(Self::Q8_0),
            GgmlType::Q2_K => Ok(Self::Q2K),
            GgmlType::Q3_K => Ok(Self::Q3K),
            GgmlType::Q4_K => Ok(Self::Q4K),
            GgmlType::Q5_K => Ok(Self::Q5K),
            GgmlType::Q6_K => Ok(Self::Q6K),
            GgmlType::IQ4_NL => Ok(Self::IQ4NL),
            GgmlType::IQ4_XS => Ok(Self::IQ4XS),
            GgmlType::IQ2_XXS => Ok(Self::IQ2XXS),
            GgmlType::IQ2_XS => Ok(Self::IQ2XS),
            GgmlType::IQ2_S => Ok(Self::IQ2S),
            GgmlType::IQ3_XXS => Ok(Self::IQ3XXS),
            GgmlType::IQ3_S => Ok(Self::IQ3S),
            GgmlType::IQ1_S => Ok(Self::IQ1S),
            GgmlType::IQ1_M => Ok(Self::IQ1M),
            GgmlType::MXFP4 => Ok(Self::MXFP4),
            _ => Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA has no kernels for GGML {ggml_type} blocks yet"
            ))),
        }
    }

    #[must_use]
    pub const fn ggml_type(self) -> GgmlType {
        match self {
            Self::Q4_0 => GgmlType::Q4_0,
            Self::Q4_1 => GgmlType::Q4_1,
            Self::Q5_0 => GgmlType::Q5_0,
            Self::Q5_1 => GgmlType::Q5_1,
            Self::Q8_0 => GgmlType::Q8_0,
            Self::Q2K => GgmlType::Q2_K,
            Self::Q3K => GgmlType::Q3_K,
            Self::Q4K => GgmlType::Q4_K,
            Self::Q5K => GgmlType::Q5_K,
            Self::Q6K => GgmlType::Q6_K,
            Self::IQ4NL => GgmlType::IQ4_NL,
            Self::IQ4XS => GgmlType::IQ4_XS,
            Self::IQ2XXS => GgmlType::IQ2_XXS,
            Self::IQ2XS => GgmlType::IQ2_XS,
            Self::IQ2S => GgmlType::IQ2_S,
            Self::IQ3XXS => GgmlType::IQ3_XXS,
            Self::IQ3S => GgmlType::IQ3_S,
            Self::IQ1S => GgmlType::IQ1_S,
            Self::IQ1M => GgmlType::IQ1_M,
            Self::MXFP4 => GgmlType::MXFP4,
        }
    }

    /// Payload bytes of `rows` rows of `columns` values.
    pub fn bytes(self, rows: usize, columns: usize) -> Result<usize> {
        let ggml_type = self.ggml_type();
        let row = ggml_type
            .row_bytes(u64::try_from(columns)?)
            .map_err(|_| Error::InvalidDecoderKernel("GGML rows must hold whole blocks"))?;
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
