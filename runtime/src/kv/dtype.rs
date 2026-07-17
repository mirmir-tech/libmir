use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheDType {
    #[default]
    Auto,
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "bfloat16")]
    BFloat16,
    #[serde(rename = "fp8")]
    Fp8,
    #[serde(rename = "fp8_e4m3")]
    Fp8E4M3,
    #[serde(rename = "fp8_e5m2")]
    Fp8E5M2,
    #[serde(rename = "fp8_inc")]
    Fp8Inc,
    #[serde(rename = "fp8_ds_mla")]
    Fp8DsMla,
    #[serde(rename = "int4_per_token_head")]
    Int4PerTokenHead,
    #[serde(rename = "int8_per_token_head")]
    Int8PerTokenHead,
    #[serde(rename = "fp8_per_token_head")]
    Fp8PerTokenHead,
    #[serde(rename = "nvfp4")]
    NvFp4,
    #[serde(rename = "turboquant_k8v4")]
    TurboQuantK8V4,
    #[serde(rename = "turboquant_4bit_nc")]
    TurboQuant4BitNc,
    #[serde(rename = "turboquant_k3v4_nc")]
    TurboQuantK3V4Nc,
    #[serde(rename = "turboquant_3bit_nc")]
    TurboQuant3BitNc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvQuantMode {
    None,
    Fp8PerTensor,
    Fp8MlaPacked,
    PerTokenHead,
    NvFp4,
    TurboQuant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvScaleGranularity {
    None,
    Tensor,
    TokenHead,
    Group,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvElementBits {
    pub key: u8,
    pub value: u8,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("unknown KV cache dtype: {0}")]
pub struct KvCacheDTypeParseError(String);

impl KvCacheDType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Fp8 => "fp8",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Fp8E5M2 => "fp8_e5m2",
            Self::Fp8Inc => "fp8_inc",
            Self::Fp8DsMla => "fp8_ds_mla",
            Self::Int4PerTokenHead => "int4_per_token_head",
            Self::Int8PerTokenHead => "int8_per_token_head",
            Self::Fp8PerTokenHead => "fp8_per_token_head",
            Self::NvFp4 => "nvfp4",
            Self::TurboQuantK8V4 => "turboquant_k8v4",
            Self::TurboQuant4BitNc => "turboquant_4bit_nc",
            Self::TurboQuantK3V4Nc => "turboquant_k3v4_nc",
            Self::TurboQuant3BitNc => "turboquant_3bit_nc",
        }
    }

    #[must_use]
    pub const fn quant_mode(self) -> KvQuantMode {
        match self {
            Self::Auto | Self::Float16 | Self::BFloat16 => KvQuantMode::None,
            Self::Fp8 | Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Fp8Inc => KvQuantMode::Fp8PerTensor,
            Self::Fp8DsMla => KvQuantMode::Fp8MlaPacked,
            Self::Int4PerTokenHead | Self::Int8PerTokenHead | Self::Fp8PerTokenHead => {
                KvQuantMode::PerTokenHead
            },
            Self::NvFp4 => KvQuantMode::NvFp4,
            Self::TurboQuantK8V4
            | Self::TurboQuant4BitNc
            | Self::TurboQuantK3V4Nc
            | Self::TurboQuant3BitNc => KvQuantMode::TurboQuant,
        }
    }

    #[must_use]
    pub const fn scale_granularity(self) -> KvScaleGranularity {
        match self {
            Self::Auto | Self::Float16 | Self::BFloat16 => KvScaleGranularity::None,
            Self::Fp8 | Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Fp8Inc => KvScaleGranularity::Tensor,
            Self::Int4PerTokenHead | Self::Int8PerTokenHead | Self::Fp8PerTokenHead => {
                KvScaleGranularity::TokenHead
            },
            Self::Fp8DsMla
            | Self::NvFp4
            | Self::TurboQuantK8V4
            | Self::TurboQuant4BitNc
            | Self::TurboQuantK3V4Nc
            | Self::TurboQuant3BitNc => KvScaleGranularity::Group,
        }
    }

    #[must_use]
    pub const fn is_quantized(self) -> bool {
        self.quant_mode().is_quantized()
    }

    #[must_use]
    pub const fn stores_native_elements(self) -> bool {
        matches!(self, Self::Auto | Self::Float16 | Self::BFloat16)
    }

    #[must_use]
    pub const fn element_bits(self, native_bits: u8) -> KvElementBits {
        match self {
            Self::Auto => KvElementBits { key: native_bits, value: native_bits },
            Self::Float16 | Self::BFloat16 => KvElementBits { key: 16, value: 16 },
            Self::Fp8
            | Self::Fp8E4M3
            | Self::Fp8E5M2
            | Self::Fp8Inc
            | Self::Fp8DsMla
            | Self::Int8PerTokenHead
            | Self::Fp8PerTokenHead => KvElementBits { key: 8, value: 8 },
            Self::Int4PerTokenHead | Self::NvFp4 | Self::TurboQuant4BitNc => {
                KvElementBits { key: 4, value: 4 }
            },
            Self::TurboQuantK8V4 => KvElementBits { key: 8, value: 4 },
            Self::TurboQuantK3V4Nc => KvElementBits { key: 3, value: 4 },
            Self::TurboQuant3BitNc => KvElementBits { key: 3, value: 3 },
        }
    }
}

impl KvQuantMode {
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl fmt::Display for KvCacheDType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for KvCacheDType {
    type Err = KvCacheDTypeParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::BFloat16),
            "fp8" => Ok(Self::Fp8),
            "fp8_e4m3" => Ok(Self::Fp8E4M3),
            "fp8_e5m2" => Ok(Self::Fp8E5M2),
            "fp8_inc" => Ok(Self::Fp8Inc),
            "fp8_ds_mla" => Ok(Self::Fp8DsMla),
            "int4_per_token_head" => Ok(Self::Int4PerTokenHead),
            "int8_per_token_head" => Ok(Self::Int8PerTokenHead),
            "fp8_per_token_head" => Ok(Self::Fp8PerTokenHead),
            "nvfp4" => Ok(Self::NvFp4),
            "turboquant_k8v4" => Ok(Self::TurboQuantK8V4),
            "turboquant_4bit_nc" => Ok(Self::TurboQuant4BitNc),
            "turboquant_k3v4_nc" => Ok(Self::TurboQuantK3V4Nc),
            "turboquant_3bit_nc" => Ok(Self::TurboQuant3BitNc),
            other => Err(KvCacheDTypeParseError(other.to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_quantized_modes() {
        assert!(!KvCacheDType::Auto.is_quantized());
        assert_eq!(KvCacheDType::Fp8E4M3.quant_mode(), KvQuantMode::Fp8PerTensor);
        assert_eq!(KvCacheDType::NvFp4.scale_granularity(), KvScaleGranularity::Group);
    }

    #[test]
    fn estimates_asymmetric_cache_bits() {
        assert_eq!(KvCacheDType::Auto.element_bits(16), KvElementBits { key: 16, value: 16 });
        assert_eq!(
            KvCacheDType::TurboQuantK8V4.element_bits(16),
            KvElementBits { key: 8, value: 4 }
        );
    }

    #[test]
    fn parses_vllm_dtype_strings() {
        assert_eq!("nvfp4".parse::<KvCacheDType>(), Ok(KvCacheDType::NvFp4));
        assert!("bogus".parse::<KvCacheDType>().is_err());
    }
}
