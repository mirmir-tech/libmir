use serde::{Deserialize, Serialize};

use crate::{
    error::{ModelsError, Result},
    weights::TensorInfo,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Eight-bit floating-point value encoding.
pub enum Float8Format {
    /// E4M3 finite-range encoding.
    E4M3,
    /// E5M2 extended-range encoding.
    E5M2,
}

impl Float8Format {
    pub(super) fn parse(tensor: &TensorInfo) -> Result<Option<Self>> {
        match tensor.dtype.as_str() {
            "F8_E4M3" => Ok(Some(Self::E4M3)),
            "F8_E5M2" => Ok(Some(Self::E5M2)),
            dtype if dtype.starts_with("F8_") => {
                Err(invalid(&tensor.name, &format!("unsupported float8 storage dtype {dtype}")))
            },
            _ => Ok(None),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::E4M3 => "F8_E4M3",
            Self::E5M2 => "F8_E5M2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// How a checkpoint scale converts stored FP8 values to model values.
pub enum Float8ScaleMode {
    /// Stored FP8 values are model values directly.
    None,
    /// Model values equal stored values multiplied by the scale.
    Multiplier,
    /// Model values equal stored values divided by the recorded inverse scale.
    InverseMultiplier,
}

impl Float8ScaleMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Multiplier => "multiplier",
            Self::InverseMultiplier => "inverse_multiplier",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Geometry of the scale tensor associated with direct FP8 values.
pub enum Float8ScaleGranularity {
    /// No weight scale is recorded.
    None,
    /// One scalar applies to the complete weight tensor.
    Tensor,
    /// One scale applies to every logical output channel.
    OutputChannel,
    /// A scale grid covers groups in the output and input dimensions.
    BlockGrid {
        /// Number of scale groups along the output dimension.
        output_groups: usize,
        /// Number of scale groups along the input dimension.
        input_groups: usize,
        /// Declared output rows per block, when checkpoint metadata provides
        /// it.
        output_block_size: Option<usize>,
        /// Declared input columns per block, when checkpoint metadata provides
        /// it.
        input_block_size: Option<usize>,
    },
}

impl Float8ScaleGranularity {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Tensor => "tensor",
            Self::OutputChannel => "output_channel",
            Self::BlockGrid { .. } => "block_grid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Floating-point dtype used by FP8 scale parameters.
pub enum Float8ParameterDType {
    /// Brain floating point.
    BF16,
    /// IEEE single precision.
    F32,
}

impl Float8ParameterDType {
    pub(super) fn parse(tensor: &TensorInfo) -> Result<Self> {
        match tensor.dtype.as_str() {
            "BF16" => Ok(Self::BF16),
            "F32" => Ok(Self::F32),
            dtype => {
                Err(invalid(&tensor.name, &format!("unsupported float8 parameter dtype {dtype}")))
            },
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BF16 => "BF16",
            Self::F32 => "F32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Float8ActivationScale {
    None,
    StaticTensor,
    DynamicToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Complete initial contract for direct FP8 projection storage.
pub struct Float8Quantization {
    pub format: Float8Format,
    pub scale_mode: Float8ScaleMode,
    pub scale_granularity: Float8ScaleGranularity,
    pub scale_dtype: Option<Float8ParameterDType>,
    pub activation_scale: Float8ActivationScale,
    pub input_scale_dtype: Option<Float8ParameterDType>,
}

impl Float8Quantization {
    #[must_use]
    pub const fn unscaled(format: Float8Format) -> Self {
        Self {
            format,
            scale_mode: Float8ScaleMode::None,
            scale_granularity: Float8ScaleGranularity::None,
            scale_dtype: None,
            activation_scale: Float8ActivationScale::None,
            input_scale_dtype: None,
        }
    }
}

fn invalid(name: &str, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid float8 binding {name}: {reason}"))
}
