use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BitsAndBytes4BitType {
    Nf4,
    Fp4,
}

impl BitsAndBytes4BitType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nf4 => "nf4",
            Self::Fp4 => "fp4",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BitsAndBytesComputeDType {
    F16,
    Bf16,
    F32,
}

impl BitsAndBytesComputeDType {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "float16" | "f16" => Some(Self::F16),
            "bfloat16" | "bf16" => Some(Self::Bf16),
            "float32" | "f32" => Some(Self::F32),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BitsAndBytesStorageDType {
    U8,
    F16,
    Bf16,
    F32,
}

impl BitsAndBytesStorageDType {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "uint8" | "u8" => Some(Self::U8),
            "float16" | "f16" => Some(Self::F16),
            "bfloat16" | "bf16" => Some(Self::Bf16),
            "float32" | "f32" => Some(Self::F32),
            _ => None,
        }
    }

    #[must_use]
    pub const fn safetensors_name(self) -> &'static str {
        match self {
            Self::U8 => "U8",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BitsAndBytes4BitQuantization {
    pub quant_type: BitsAndBytes4BitType,
    pub block_size: usize,
    pub compute_dtype: BitsAndBytesComputeDType,
    pub storage_dtype: BitsAndBytesStorageDType,
    pub nested_block_size: Option<usize>,
}

impl BitsAndBytes4BitQuantization {
    #[must_use]
    pub const fn is_supported(self) -> bool {
        self.block_size == 64
            && matches!(self.compute_dtype, BitsAndBytesComputeDType::Bf16)
            && matches!(self.nested_block_size, None | Some(256))
    }
}
