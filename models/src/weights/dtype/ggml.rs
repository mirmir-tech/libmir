//! GGML tensor types as GGUF files store them. Block geometry is taken from
//! llama.cpp v1.0.0 (`ggml-common.h`, `type_traits` in `ggml.c`); the sizes
//! are checked against those structures by `tests::block_sizes_match_ggml`.

use crate::error::{ModelsError, Result};

/// A tensor type identifier of the GGUF tensor directory. Removed GGML types
/// (4, 5 and the runtime-repacked 31–33, 36–38) are refused by number.
#[expect(non_camel_case_types, reason = "variants spell GGML's own type names")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    F64,
    F8_E4M3,
    I8,
    I16,
    I32,
    I64,
    Q1_0,
    Q2_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_NL,
    IQ4_XS,
    TQ1_0,
    TQ2_0,
    MXFP4,
    NVFP4,
}

/// (type, GGUF id, canonical name, elements per block, bytes per block)
const TABLE: [(GgmlType, u32, &str, u64, u64); 36] = [
    (GgmlType::F32, 0, "F32", 1, 4),
    (GgmlType::F16, 1, "F16", 1, 2),
    (GgmlType::Q4_0, 2, "Q4_0", 32, 18),
    (GgmlType::Q4_1, 3, "Q4_1", 32, 20),
    (GgmlType::Q5_0, 6, "Q5_0", 32, 22),
    (GgmlType::Q5_1, 7, "Q5_1", 32, 24),
    (GgmlType::Q8_0, 8, "Q8_0", 32, 34),
    (GgmlType::Q8_1, 9, "Q8_1", 32, 36),
    (GgmlType::Q2_K, 10, "Q2_K", 256, 84),
    (GgmlType::Q3_K, 11, "Q3_K", 256, 110),
    (GgmlType::Q4_K, 12, "Q4_K", 256, 144),
    (GgmlType::Q5_K, 13, "Q5_K", 256, 176),
    (GgmlType::Q6_K, 14, "Q6_K", 256, 210),
    (GgmlType::Q8_K, 15, "Q8_K", 256, 292),
    (GgmlType::IQ2_XXS, 16, "IQ2_XXS", 256, 66),
    (GgmlType::IQ2_XS, 17, "IQ2_XS", 256, 74),
    (GgmlType::IQ3_XXS, 18, "IQ3_XXS", 256, 98),
    (GgmlType::IQ1_S, 19, "IQ1_S", 256, 50),
    (GgmlType::IQ4_NL, 20, "IQ4_NL", 32, 18),
    (GgmlType::IQ3_S, 21, "IQ3_S", 256, 110),
    (GgmlType::IQ2_S, 22, "IQ2_S", 256, 82),
    (GgmlType::IQ4_XS, 23, "IQ4_XS", 256, 136),
    (GgmlType::I8, 24, "I8", 1, 1),
    (GgmlType::I16, 25, "I16", 1, 2),
    (GgmlType::I32, 26, "I32", 1, 4),
    (GgmlType::I64, 27, "I64", 1, 8),
    (GgmlType::F64, 28, "F64", 1, 8),
    (GgmlType::IQ1_M, 29, "IQ1_M", 256, 56),
    (GgmlType::BF16, 30, "BF16", 1, 2),
    (GgmlType::TQ1_0, 34, "TQ1_0", 256, 54),
    (GgmlType::TQ2_0, 35, "TQ2_0", 256, 66),
    (GgmlType::MXFP4, 39, "MXFP4", 32, 17),
    (GgmlType::NVFP4, 40, "NVFP4", 64, 36),
    (GgmlType::Q1_0, 41, "Q1_0", 128, 18),
    (GgmlType::Q2_0, 42, "Q2_0", 64, 18),
    (GgmlType::F8_E4M3, 43, "F8_E4M3", 1, 1),
];

impl GgmlType {
    pub fn from_id(id: u32) -> Result<Self> {
        TABLE
            .iter()
            .find(|entry| entry.1 == id)
            .map(|entry| entry.0)
            .ok_or(ModelsError::UnsupportedGgmlType(id))
    }

    fn entry(self) -> &'static (Self, u32, &'static str, u64, u64) {
        // Every variant has exactly one row; `every_type_has_one_row` checks it.
        TABLE.iter().find(|entry| entry.0 == self).unwrap_or(&TABLE[0])
    }

    /// The type a catalog dtype names, by its canonical name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        TABLE.iter().find(|entry| entry.2 == name).map(|entry| entry.0)
    }

    #[must_use]
    pub fn id(self) -> u32 {
        self.entry().1
    }

    /// The name as `gguf-dump` and model cards print it, for example `Q4_K`.
    #[must_use]
    pub fn name(self) -> &'static str {
        self.entry().2
    }

    #[must_use]
    pub fn block_elements(self) -> u64 {
        self.entry().3
    }

    #[must_use]
    pub fn block_bytes(self) -> u64 {
        self.entry().4
    }

    /// Bytes for one row of `elements` values; a row must hold whole blocks.
    pub fn row_bytes(self, elements: u64) -> Result<u64> {
        let block = self.block_elements();
        if !elements.is_multiple_of(block) {
            return Err(ModelsError::InvalidGguf(format!(
                "{} rows hold whole blocks of {block} values; found a row of {elements}",
                self.name()
            )));
        }
        (elements / block)
            .checked_mul(self.block_bytes())
            .ok_or_else(|| ModelsError::InvalidGguf("tensor size overflows".into()))
    }

    #[must_use]
    pub fn bits_per_weight(self) -> f64 {
        // Block sizes are below 2^16, exact in f64.
        let bytes = u32::try_from(self.block_bytes()).unwrap_or(u32::MAX);
        let elements = u32::try_from(self.block_elements()).unwrap_or(1);
        f64::from(bytes) * 8.0 / f64::from(elements)
    }

    /// Plain floating-point or integer storage, one value per element.
    #[must_use]
    pub fn is_dense(self) -> bool {
        self.block_elements() == 1
    }

    /// The type as affine groups of 32 values, `scale * u + bias` with
    /// unsigned `u`, when each of its 32-value sub-blocks shares one scale
    /// and minimum; `None` for finer scales (`Q2_K`, `Q3_K`, `Q6_K`) and for
    /// lookup-table or non-uniform codes.
    #[must_use]
    pub const fn affine32(self) -> Option<Affine32> {
        let (bits, offset) = match self {
            Self::Q4_0 => (4, 8),
            Self::Q4_1 | Self::Q4_K => (4, 0),
            Self::Q5_0 => (5, 16),
            Self::Q5_1 | Self::Q5_K => (5, 0),
            Self::Q8_0 => (8, 128),
            _ => return None,
        };
        Some(Affine32 { bits, offset })
    }

    pub fn all() -> impl Iterator<Item = Self> {
        TABLE.iter().map(|entry| entry.0)
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// A GGML type's affine groups of 32: `bits` per value, and the `offset` that
/// moves its signed integers onto `[0, 2^bits)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Affine32 {
    pub bits: u8,
    pub offset: u8,
}

impl Affine32 {
    /// Whether Metal keeps the type as affine groups. MLX's 4-bit products
    /// match or beat the GGML GEMV on one row and stay near it on four (`Q4_K`
    /// 4096 to 12288: 104/128/164 us for 1/2/4 rows as GGML, 105/112/114 as
    /// affine); 5- and 8-bit ones are slower on one row (`Q5_0` 116 vs 136 us,
    /// `Q8_0` 188 vs 194 us), so those types keep their GGML blocks.
    #[must_use]
    pub const fn repacks_on_metal(self) -> bool {
        self.bits == 4
    }

    /// Bytes of `elements` values as affine groups with 16-bit scales and
    /// biases.
    #[must_use]
    pub const fn bytes(self, elements: u64) -> u64 {
        elements * self.bits as u64 / 8 + elements / 32 * 4
    }
}
