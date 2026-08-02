use serde::{Deserialize, Serialize};

use crate::{
    error::{ModelsError, Result},
    weights::TensorInfo,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GptqBits {
    Two,
    Three,
    Four,
    Eight,
}

impl GptqBits {
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

impl TryFrom<u8> for GptqBits {
    type Error = ModelsError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            8 => Ok(Self::Eight),
            _ => Err(ModelsError::InvalidConfig(format!("unsupported GPTQ width {value}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GptqCheckpointFormat {
    Gptq,
    GptqV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GptqPacking {
    InputLittleEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GptqStorageDType {
    I32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GptqScaleDType {
    F16,
    BF16,
}

impl GptqScaleDType {
    pub(super) fn parse(tensor: &TensorInfo) -> Result<Self> {
        match tensor.dtype.as_str() {
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            _ => Err(ModelsError::InvalidConfig(format!(
                "GPTQ scales must use F16 or BF16: {}",
                tensor.name
            ))),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GptqQuantization {
    pub bits: GptqBits,
    pub group_size: usize,
    pub packing: GptqPacking,
    pub storage_dtype: GptqStorageDType,
    pub scale_dtype: GptqScaleDType,
    pub checkpoint_format: GptqCheckpointFormat,
    pub symmetric: bool,
    pub activation_order: bool,
    pub packed_zero_points: bool,
}

impl GptqQuantization {
    #[must_use]
    pub const fn is_input_packed(self) -> bool {
        self.group_size > 0
            && matches!(self.packing, GptqPacking::InputLittleEndian)
            && matches!(self.storage_dtype, GptqStorageDType::I32)
            && self.packed_zero_points
    }
}
