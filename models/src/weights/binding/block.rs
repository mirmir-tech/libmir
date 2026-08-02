use serde::{Deserialize, Serialize};

use super::{BindingTransform, TensorBinding, TensorPacking, TensorStorage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Numerical values represented by one block-quantized weight payload.
pub enum BlockFormat {
    /// OCP MX four-bit floating point.
    MxFp4,
    /// OCP MX eight-bit floating point.
    MxFp8,
    /// NVIDIA four-bit floating point.
    NvFp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Physical `SafeTensors` dtype of values or scale parameters.
pub enum BlockStorageDType {
    /// Packed byte storage.
    U8,
    /// Packed 32-bit word storage.
    U32,
    /// Eight-bit E4M3 floating-point storage.
    F8E4M3,
    /// IEEE single precision.
    F32,
    /// Brain floating-point storage.
    BF16,
}

impl BlockStorageDType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "U8",
            Self::U32 => "U32",
            Self::F8E4M3 => "F8_E4M3",
            Self::F32 => "F32",
            Self::BF16 => "BF16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Numerical interpretation of a block scale.
pub enum BlockScaleEncoding {
    /// OCP E8M0 exponent-only scale.
    E8M0,
    /// Signed E4M3 floating-point scale.
    F8E4M3,
    /// IEEE single-precision scale.
    F32,
}

impl BlockScaleEncoding {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::E8M0 => "E8M0",
            Self::F8E4M3 => "F8_E4M3",
            Self::F32 => "F32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Typed storage and numerical interpretation of one scale level.
pub struct BlockScale {
    pub encoding: BlockScaleEncoding,
    pub storage_dtype: BlockStorageDType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Whether physical input width may exceed the logical projection width.
pub enum BlockInputPadding {
    /// Every logical input dimension must divide the physical block geometry.
    Forbidden,
    /// Physical storage may pad the input axis to a complete block.
    ToBlock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Activation contract paired with a block-quantized checkpoint weight.
pub enum BlockActivationMode {
    /// Activations remain in the model compute dtype.
    WeightOnly,
    /// Activations are quantized to the checkpoint block format at execution.
    WeightAndActivation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Complete physical contract for block floating-point storage.
pub struct BlockQuantization {
    pub format: BlockFormat,
    pub block_size: usize,
    pub storage_dtype: BlockStorageDType,
    pub block_scale: BlockScale,
    pub global_scale: Option<BlockScale>,
    pub input_scale: Option<BlockScale>,
    pub output_bias_dtype: Option<BlockStorageDType>,
    pub input_padding: BlockInputPadding,
    pub activation_mode: BlockActivationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Logical matrix organization retained independently from block encoding.
pub enum BlockProjectionLayout {
    Matrix,
    MatrixBank { matrices: usize },
    FusedGateUpBank { experts: usize, interleaved: bool },
}

impl BlockQuantization {
    pub const MXFP4: Self = Self {
        format: BlockFormat::MxFp4,
        block_size: 32,
        storage_dtype: BlockStorageDType::U8,
        block_scale: BlockScale {
            encoding: BlockScaleEncoding::E8M0,
            storage_dtype: BlockStorageDType::U8,
        },
        global_scale: None,
        input_scale: None,
        output_bias_dtype: Some(BlockStorageDType::BF16),
        input_padding: BlockInputPadding::Forbidden,
        activation_mode: BlockActivationMode::WeightOnly,
    };
    /// MLX container for the same MXFP4 bytes packed into U32 words.
    pub const MXFP4_MLX: Self = Self {
        storage_dtype: BlockStorageDType::U32,
        ..Self::MXFP4
    };
    pub const MXFP8: Self = Self {
        format: BlockFormat::MxFp8,
        block_size: 32,
        storage_dtype: BlockStorageDType::U32,
        block_scale: BlockScale {
            encoding: BlockScaleEncoding::E8M0,
            storage_dtype: BlockStorageDType::U8,
        },
        global_scale: None,
        input_scale: None,
        output_bias_dtype: Some(BlockStorageDType::BF16),
        input_padding: BlockInputPadding::Forbidden,
        activation_mode: BlockActivationMode::WeightOnly,
    };
    pub const NVFP4: Self = Self {
        format: BlockFormat::NvFp4,
        block_size: 16,
        storage_dtype: BlockStorageDType::U8,
        block_scale: BlockScale {
            encoding: BlockScaleEncoding::F8E4M3,
            storage_dtype: BlockStorageDType::F8E4M3,
        },
        global_scale: Some(BlockScale {
            encoding: BlockScaleEncoding::F32,
            storage_dtype: BlockStorageDType::F32,
        }),
        input_scale: Some(BlockScale {
            encoding: BlockScaleEncoding::F32,
            storage_dtype: BlockStorageDType::F32,
        }),
        output_bias_dtype: None,
        input_padding: BlockInputPadding::Forbidden,
        activation_mode: BlockActivationMode::WeightAndActivation,
    };
    pub const NVFP4_W4A16: Self = Self {
        activation_mode: BlockActivationMode::WeightOnly,
        ..Self::NVFP4
    };

    #[must_use]
    pub const fn is_mxfp4(self) -> bool {
        matches!(self.format, BlockFormat::MxFp4)
            && self.block_size == Self::MXFP4.block_size
            && matches!(self.storage_dtype, BlockStorageDType::U8 | BlockStorageDType::U32)
            && matches!(self.block_scale.encoding, BlockScaleEncoding::E8M0)
            && matches!(self.block_scale.storage_dtype, BlockStorageDType::U8)
            && self.global_scale.is_none()
            && self.input_scale.is_none()
            && matches!(self.output_bias_dtype, Some(BlockStorageDType::BF16))
            && matches!(self.input_padding, BlockInputPadding::Forbidden)
    }
}

impl TensorBinding {
    #[must_use]
    pub fn block_projection_layout(&self) -> Option<BlockProjectionLayout> {
        let TensorStorage::BlockQuantized { packing, .. } = self.storage else {
            return None;
        };
        let mut matrices = None;
        let mut fused = None;
        for transform in &self.transforms {
            match *transform {
                BindingTransform::StackedExperts { count } if matrices.replace(count).is_none() => {
                },
                BindingTransform::FusedGateUp { interleaved }
                    if fused.replace(interleaved).is_none() => {},
                BindingTransform::Transpose
                | BindingTransform::FusedQkv { .. }
                | BindingTransform::StackedExperts { .. }
                | BindingTransform::FusedGateUp { .. } => return None,
            }
        }
        match (matrices, fused, packing) {
            (None, None, TensorPacking::Separate) => Some(BlockProjectionLayout::Matrix),
            (Some(matrices), None, TensorPacking::Separate) => {
                Some(BlockProjectionLayout::MatrixBank { matrices })
            },
            (Some(experts), Some(false), TensorPacking::Separate) => {
                Some(BlockProjectionLayout::FusedGateUpBank { experts, interleaved: false })
            },
            (Some(experts), Some(true), TensorPacking::InterleavedGateUp) => {
                Some(BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true })
            },
            _ => None,
        }
    }
}
