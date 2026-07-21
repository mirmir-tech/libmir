use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendTarget {
    Cuda,
    Metal,
    CpuReference,
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
    MxFp4,
    Custom(String),
}

impl Quantization {
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        match self {
            Self::Int4 | Self::Int8 | Self::Fp8 | Self::NvFp4 | Self::MxFp4 => true,
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
    pub id: String,
    pub path: String,
    pub tokenizer_path: Option<String>,
    pub context_len: usize,
    pub quantization: Quantization,
    pub preferred_backends: Vec<BackendTarget>,
}
