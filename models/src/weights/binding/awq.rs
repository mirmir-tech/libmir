use serde::{Deserialize, Serialize};

use crate::{
    error::{ModelsError, Result},
    weights::TensorInfo,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Bit width supported by the `AutoAWQ` GEMM checkpoint contract.
pub enum AwqBits {
    Four,
}

impl AwqBits {
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::Four => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AwqPacking {
    /// Input-major words packing eight interleaved output nibbles.
    GemmOutputInterleaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AwqStorageDType {
    I32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AwqScaleDType {
    F16,
}

impl AwqScaleDType {
    pub(super) fn parse(tensor: &TensorInfo) -> Result<Self> {
        if tensor.dtype == "F16" {
            Ok(Self::F16)
        } else {
            Err(ModelsError::InvalidConfig(format!("AWQ scales must use F16: {}", tensor.name)))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Complete `AutoAWQ` GEMM W4A16 asymmetric group contract.
pub struct AwqQuantization {
    pub bits: AwqBits,
    pub group_size: usize,
    pub packing: AwqPacking,
    pub storage_dtype: AwqStorageDType,
    pub scale_dtype: AwqScaleDType,
    pub packed_zero_points: bool,
}

impl AwqQuantization {
    #[must_use]
    pub const fn is_gemm_w4a16(self) -> bool {
        matches!(self.bits, AwqBits::Four)
            && self.group_size > 0
            && matches!(self.packing, AwqPacking::GemmOutputInterleaved)
            && matches!(self.storage_dtype, AwqStorageDType::I32)
            && matches!(self.scale_dtype, AwqScaleDType::F16)
            && self.packed_zero_points
    }
}
