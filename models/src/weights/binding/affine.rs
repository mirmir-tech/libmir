use serde::{Deserialize, Serialize};

use crate::{
    error::{ModelsError, Result},
    weights::TensorInfo,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
/// Supported bit widths in native MLX grouped-affine storage.
pub enum AffineBits {
    /// Two bits per logical value.
    Two,
    /// Three bits per logical value.
    Three,
    /// Four bits per logical value.
    Four,
    /// Five bits per logical value.
    Five,
    /// Six bits per logical value.
    Six,
    /// Eight bits per logical value.
    Eight,
}

impl AffineBits {
    #[must_use]
    /// Returns the numeric bit width.
    pub const fn get(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Five => 5,
            Self::Six => 6,
            Self::Eight => 8,
        }
    }
}

impl From<AffineBits> for u8 {
    fn from(value: AffineBits) -> Self {
        value.get()
    }
}

impl TryFrom<u8> for AffineBits {
    type Error = ModelsError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            5 => Ok(Self::Five),
            6 => Ok(Self::Six),
            8 => Ok(Self::Eight),
            _ => Err(ModelsError::InvalidConfig(format!(
                "unsupported grouped-affine bit width {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Interpretation of each unpacked integer value.
pub enum AffineSignedness {
    /// Values are non-negative integers.
    Unsigned,
    /// Values use a signed integer representation.
    Signed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// How an affine group records its zero-point term.
pub enum AffineZeroPointMode {
    /// Dequantization applies only the scale.
    None,
    /// Dequantization adds a per-group floating-point bias.
    AdditiveBias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Logical tensor axis partitioned into quantization groups.
pub enum AffineGroupAxis {
    /// Groups partition projection input features.
    Input,
    /// Groups partition projection output features.
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Bit ordering used by a packed grouped-affine tensor.
pub enum AffinePacking {
    /// Native MLX low-bit ordering.
    Mlx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Safetensors container dtype holding packed integer values.
pub enum AffineStorageDType {
    /// Byte-addressed packed storage.
    U8,
    /// 32-bit-word packed storage.
    U32,
}

impl AffineStorageDType {
    pub(super) fn parse(source: &TensorInfo) -> Result<Self> {
        match source.dtype.as_str() {
            "U8" => Ok(Self::U8),
            "U32" => Ok(Self::U32),
            dtype => Err(invalid(&source.name, &format!("unsupported packed dtype {dtype}"))),
        }
    }

    #[must_use]
    /// Returns the number of storage bits in one physical element.
    pub const fn bits(self) -> usize {
        match self {
            Self::U8 => 8,
            Self::U32 => 32,
        }
    }

    #[must_use]
    /// Returns the canonical `SafeTensors` dtype spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "U8",
            Self::U32 => "U32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Floating-point dtype of per-group affine parameters.
pub enum AffineParameterDType {
    /// IEEE half precision.
    F16,
    /// Brain floating point.
    BF16,
    /// IEEE single precision.
    F32,
}

impl AffineParameterDType {
    pub(super) fn parse(tensor: &TensorInfo) -> Result<Self> {
        match tensor.dtype.as_str() {
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "F32" => Ok(Self::F32),
            dtype => Err(invalid(&tensor.name, &format!("unsupported parameter dtype {dtype}"))),
        }
    }

    #[must_use]
    /// Returns the canonical `SafeTensors` dtype spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Complete physical contract for one grouped-affine tensor binding.
pub struct GroupedAffineQuantization {
    /// Number of packed bits per logical value.
    pub bits: AffineBits,
    /// Logical values sharing one scale and optional bias.
    pub group_size: usize,
    /// Axis partitioned into groups.
    pub group_axis: AffineGroupAxis,
    /// Signed or unsigned integer interpretation.
    pub signedness: AffineSignedness,
    /// Zero-point representation.
    pub zero_point: AffineZeroPointMode,
    /// Packed bit ordering.
    pub packing: AffinePacking,
    /// Physical packed storage dtype.
    pub storage_dtype: AffineStorageDType,
    /// Per-group scale dtype.
    pub scale_dtype: AffineParameterDType,
    /// Per-group bias dtype when an additive bias is present.
    pub bias_dtype: Option<AffineParameterDType>,
}

impl GroupedAffineQuantization {
    #[must_use]
    /// Reports whether integer interpretation, grouping, and packing match MLX.
    pub fn is_mlx_layout(self) -> bool {
        self.group_size > 0
            && self.group_axis == AffineGroupAxis::Input
            && self.signedness == AffineSignedness::Unsigned
            && self.packing == AffinePacking::Mlx
    }

    #[must_use]
    /// Reports whether the declared additive bias has a matching dtype.
    pub fn has_additive_bias(self) -> bool {
        self.zero_point == AffineZeroPointMode::AdditiveBias
            && self.bias_dtype == Some(self.scale_dtype)
    }
}

fn invalid(name: &str, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid grouped-affine binding {name}: {reason}"))
}
