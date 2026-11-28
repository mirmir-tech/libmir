//! The family of a weight encoding without its parameters: what a Hub tag
//! names before the checkpoint is inspected.

use super::WeightEncoding;

/// The family of a [`WeightEncoding`], without the parameters that complete
/// its physical contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EncodingKind {
    /// Ordinary floating-point tensor storage.
    Dense,
    /// Packed grouped affine integer storage.
    Affine,
    /// Compressed-tensors packed integer storage.
    PackedInteger,
    /// `AutoAWQ` GEMM storage.
    Awq,
    /// GPTQ input-packed storage.
    Gptq,
    /// bitsandbytes NF4 or FP4 storage.
    BitsAndBytes4Bit,
    /// Direct FP8 projection storage.
    Float8,
    /// OCP MXFP4 blocks.
    MxFp4,
    /// OCP MXFP8 blocks.
    MxFp8,
    /// NVIDIA NVFP4 blocks.
    NvFp4,
    /// GGML blocks from GGUF files.
    Ggml,
}

impl EncodingKind {
    /// The family's display name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Affine => "Grouped affine",
            Self::PackedInteger => "Compressed integer",
            Self::Awq => "AWQ",
            Self::Gptq => "GPTQ",
            Self::BitsAndBytes4Bit => "bitsandbytes",
            Self::Float8 => "FP8",
            Self::MxFp4 => "MXFP4",
            Self::MxFp8 => "MXFP8",
            Self::NvFp4 => "NVFP4",
            Self::Ggml => "GGML",
        }
    }
}

impl WeightEncoding {
    /// The family of this encoding.
    #[must_use]
    pub const fn kind(&self) -> EncodingKind {
        match self {
            Self::Dense { .. } => EncodingKind::Dense,
            Self::Affine { .. } => EncodingKind::Affine,
            Self::PackedInteger { .. } => EncodingKind::PackedInteger,
            Self::Awq { .. } => EncodingKind::Awq,
            Self::Gptq { .. } => EncodingKind::Gptq,
            Self::BitsAndBytes4Bit { .. } => EncodingKind::BitsAndBytes4Bit,
            Self::Float8 { .. } => EncodingKind::Float8,
            Self::MxFp4 { .. } => EncodingKind::MxFp4,
            Self::MxFp8 { .. } => EncodingKind::MxFp8,
            Self::NvFp4 { .. } => EncodingKind::NvFp4,
            Self::Ggml { .. } => EncodingKind::Ggml,
        }
    }
}
