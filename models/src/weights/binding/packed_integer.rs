use serde::{Deserialize, Serialize};

use crate::{
    error::{ModelsError, Result},
    weights::TensorInfo,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
/// Bit width of a compressed-tensors packed integer value.
pub enum CompressedIntegerBits {
    /// Four-bit weight-only integer storage.
    Four,
    /// Eight-bit weight-only integer storage.
    Eight,
}

impl CompressedIntegerBits {
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

impl From<CompressedIntegerBits> for u8 {
    fn from(value: CompressedIntegerBits) -> Self {
        value.get()
    }
}

impl TryFrom<u8> for CompressedIntegerBits {
    type Error = ModelsError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            4 => Ok(Self::Four),
            8 => Ok(Self::Eight),
            _ => Err(ModelsError::InvalidConfig(format!(
                "unsupported compressed integer bit width {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressedIntegerScaleStrategy {
    /// One static scale per output row.
    Channel,
    /// One static scale per input group.
    Group {
        /// Number of consecutive input values sharing a scale.
        group_size: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressedIntegerSignedness {
    /// Signed values shifted by `2^(bits - 1)` before packing.
    OffsetBinary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressedIntegerZeroPointMode {
    /// Symmetric quantization; the checkpoint stores no zero point.
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressedIntegerActivationOrder {
    /// Columns retain their logical order and no group-index tensor exists.
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressedIntegerPacking {
    /// Dense little-endian bitstream along the projection input axis.
    DenseLittleEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CompressedIntegerStorageDType {
    /// Signed 32-bit words used only as a packed bit container.
    I32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CompressedIntegerScaleDType {
    F16,
    BF16,
    F32,
}

impl CompressedIntegerScaleDType {
    pub(super) fn parse(tensor: &TensorInfo) -> Result<Self> {
        match tensor.dtype.as_str() {
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "F32" => Ok(Self::F32),
            dtype => Err(invalid(&tensor.name, &format!("unsupported scale dtype {dtype}"))),
        }
    }

    #[must_use]
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
/// Complete contract for symmetric per-channel compressed-tensors INT8 storage.
pub struct CompressedIntegerQuantization {
    pub bits: CompressedIntegerBits,
    pub scale_strategy: CompressedIntegerScaleStrategy,
    pub signedness: CompressedIntegerSignedness,
    pub zero_point: CompressedIntegerZeroPointMode,
    pub activation_order: CompressedIntegerActivationOrder,
    pub packing: CompressedIntegerPacking,
    pub storage_dtype: CompressedIntegerStorageDType,
    pub scale_dtype: CompressedIntegerScaleDType,
}

impl CompressedIntegerQuantization {
    #[must_use]
    pub fn is_symmetric_channel_int8(self) -> bool {
        self.bits == CompressedIntegerBits::Eight
            && self.scale_strategy == CompressedIntegerScaleStrategy::Channel
            && self.signedness == CompressedIntegerSignedness::OffsetBinary
            && self.zero_point == CompressedIntegerZeroPointMode::None
            && self.activation_order == CompressedIntegerActivationOrder::None
            && self.packing == CompressedIntegerPacking::DenseLittleEndian
            && self.storage_dtype == CompressedIntegerStorageDType::I32
    }

    #[must_use]
    pub fn is_symmetric_group_int4(self) -> bool {
        self.bits == CompressedIntegerBits::Four
            && matches!(
                self.scale_strategy,
                CompressedIntegerScaleStrategy::Group { group_size } if group_size > 0
            )
            && self.signedness == CompressedIntegerSignedness::OffsetBinary
            && self.zero_point == CompressedIntegerZeroPointMode::None
            && self.activation_order == CompressedIntegerActivationOrder::None
            && self.packing == CompressedIntegerPacking::DenseLittleEndian
            && self.storage_dtype == CompressedIntegerStorageDType::I32
    }
}

fn invalid(name: &str, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid compressed integer binding {name}: {reason}"))
}
