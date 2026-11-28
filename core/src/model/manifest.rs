use std::{borrow::Borrow, fmt};

use serde::{Deserialize, Serialize};

/// Stable identifier of a loaded model across runtime, KV cache, and backends.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ModelId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for ModelId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<String> for ModelId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for ModelId {
    fn from(id: &str) -> Self {
        Self(id.to_owned())
    }
}

impl From<&String> for ModelId {
    fn from(id: &String) -> Self {
        Self(id.clone())
    }
}

impl From<&Self> for ModelId {
    fn from(id: &Self) -> Self {
        id.clone()
    }
}

impl From<ModelId> for String {
    fn from(id: ModelId) -> Self {
        id.0
    }
}

impl PartialEq<str> for ModelId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for ModelId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BackendTarget {
    Cuda,
    Metal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    None,
    F16,
    Bf16,
    Int8,
    Int4,
    Fp8,
    NvFp4,
    Nf4,
    Fp4,
    MxFp4,
    MxFp8,
    Custom(String),
}

impl Quantization {
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        match self {
            Self::Int4
            | Self::Int8
            | Self::Fp8
            | Self::NvFp4
            | Self::Nf4
            | Self::Fp4
            | Self::MxFp4
            | Self::MxFp8 => true,
            Self::Custom(kind) => {
                let kind = kind.to_ascii_lowercase();
                kind.contains("bit")
                    || kind.contains("int")
                    || kind.contains("fp8")
                    || kind.contains("quant")
            },
            Self::None | Self::F16 | Self::Bf16 => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub id: ModelId,
    pub path: String,
    pub tokenizer_path: Option<String>,
    pub context_len: usize,
    pub quantization: Quantization,
    pub preferred_backends: Vec<BackendTarget>,
}
