use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Floating-point dtypes admitted for dense compute weights and decision plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatDType {
    F16,
    Bf16,
    F32,
}

impl FloatDType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }
}

impl fmt::Display for FloatDType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Element-wise dense tensor storage dtype (non-block-quantized).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DenseDType {
    Bf16,
    Bool,
    F16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Other(String),
}

impl DenseDType {
    /// Parses a known `SafeTensors` or dense GGML dtype spelling.
    #[must_use]
    pub fn parse_known(name: &str) -> Option<Self> {
        match name.to_ascii_uppercase().as_str() {
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            "F32" => Some(Self::F32),
            "F64" => Some(Self::F64),
            "F8_E4M3" => Some(Self::F8E4M3),
            "F8_E5M2" => Some(Self::F8E5M2),
            "U8" => Some(Self::U8),
            "I8" => Some(Self::I8),
            "U16" => Some(Self::U16),
            "I16" => Some(Self::I16),
            "U32" => Some(Self::U32),
            "I32" => Some(Self::I32),
            "U64" => Some(Self::U64),
            "I64" => Some(Self::I64),
            "BOOL" => Some(Self::Bool),
            _ => None,
        }
    }

    #[must_use]
    pub fn parse(name: &str) -> Self {
        Self::parse_known(name).unwrap_or_else(|| Self::Other(name.to_owned()))
    }

    #[must_use]
    pub const fn as_float(&self) -> Option<FloatDType> {
        match self {
            Self::F16 => Some(FloatDType::F16),
            Self::Bf16 => Some(FloatDType::Bf16),
            Self::F32 => Some(FloatDType::F32),
            _ => None,
        }
    }

    #[must_use]
    pub const fn is_float(&self) -> bool {
        self.as_float().is_some()
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::U16 => "U16",
            Self::I16 => "I16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::U64 => "U64",
            Self::I64 => "I64",
            Self::Bool => "BOOL",
            Self::Other(name) => name,
        }
    }
}

impl From<FloatDType> for DenseDType {
    fn from(dtype: FloatDType) -> Self {
        match dtype {
            FloatDType::F16 => Self::F16,
            FloatDType::Bf16 => Self::Bf16,
            FloatDType::F32 => Self::F32,
        }
    }
}

impl fmt::Display for DenseDType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for DenseDType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DenseDType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(Self::parse(&name))
    }
}
