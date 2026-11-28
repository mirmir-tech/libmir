//! Typed checkpoint tensor dtypes: parsed once at the container boundary so
//! binding, admission, and backend loaders match on enums rather than strings.

mod dense;
mod ggml;

use std::fmt;

pub use dense::{DenseDType, FloatDType};
pub use ggml::{Affine32, GgmlType};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The storage dtype of a cataloged tensor: either element-wise dense or
/// block-quantized GGML.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorDType {
    Dense(DenseDType),
    Ggml(GgmlType),
}

impl TensorDType {
    /// Parses a dtype string from a `SafeTensors` header or fixture.
    #[must_use]
    pub fn parse(name: &str) -> Self {
        if let Some(dense) = DenseDType::parse_known(name) {
            return Self::Dense(dense);
        }
        if let Some(ggml) = GgmlType::from_name(name).filter(|kind| !kind.is_dense()) {
            return Self::Ggml(ggml);
        }
        Self::Dense(DenseDType::Other(name.to_owned()))
    }

    /// Converts a GGUF tensor type into its catalog dtype without stringifying:
    /// dense GGML types map to [`Self::Dense`], quantized types to
    /// [`Self::Ggml`].
    #[must_use]
    pub const fn from_ggml(ggml: GgmlType) -> Self {
        match ggml {
            GgmlType::F32 => Self::Dense(DenseDType::F32),
            GgmlType::F16 => Self::Dense(DenseDType::F16),
            GgmlType::BF16 => Self::Dense(DenseDType::Bf16),
            GgmlType::F64 => Self::Dense(DenseDType::F64),
            GgmlType::F8_E4M3 => Self::Dense(DenseDType::F8E4M3),
            GgmlType::I8 => Self::Dense(DenseDType::I8),
            GgmlType::I16 => Self::Dense(DenseDType::I16),
            GgmlType::I32 => Self::Dense(DenseDType::I32),
            GgmlType::I64 => Self::Dense(DenseDType::I64),
            quantized => Self::Ggml(quantized),
        }
    }

    #[must_use]
    pub const fn as_dense(&self) -> Option<&DenseDType> {
        match self {
            Self::Dense(dtype) => Some(dtype),
            Self::Ggml(_) => None,
        }
    }

    #[must_use]
    pub const fn as_float(&self) -> Option<FloatDType> {
        match self {
            Self::Dense(dtype) => dtype.as_float(),
            Self::Ggml(_) => None,
        }
    }

    #[must_use]
    pub const fn as_ggml(&self) -> Option<GgmlType> {
        match self {
            Self::Ggml(ggml) => Some(*ggml),
            Self::Dense(_) => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Dense(dtype) => dtype.as_str(),
            Self::Ggml(ggml) => ggml.name(),
        }
    }
}

impl From<DenseDType> for TensorDType {
    fn from(dtype: DenseDType) -> Self {
        Self::Dense(dtype)
    }
}

impl From<FloatDType> for TensorDType {
    fn from(dtype: FloatDType) -> Self {
        Self::Dense(dtype.into())
    }
}

impl PartialEq<DenseDType> for TensorDType {
    fn eq(&self, other: &DenseDType) -> bool {
        matches!(self, Self::Dense(dtype) if dtype == other)
    }
}

impl fmt::Display for TensorDType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for TensorDType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for TensorDType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(Self::parse(&name))
    }
}
