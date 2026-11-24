//! GGML block types each backend executes in place, declared once for
//! admission and for the backends' kernel tables, which test against it.

use models::gguf::GgmlType;

/// The 25 quantized GGML types with kernels on both backends.
const BLOCK_TYPES: [GgmlType; 25] = [
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
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_XS,
    GgmlType::IQ2_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ3_S,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
    GgmlType::MXFP4,
    GgmlType::NVFP4,
    GgmlType::Q1_0,
    GgmlType::Q2_0,
    GgmlType::TQ1_0,
    GgmlType::TQ2_0,
];

/// GGML block types with CUDA kernels.
pub const CUDA: &[GgmlType] = &BLOCK_TYPES;

/// GGML block types with Metal kernels.
pub const METAL: &[GgmlType] = &BLOCK_TYPES;
