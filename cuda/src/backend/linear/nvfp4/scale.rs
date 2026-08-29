use crate::{Error, Result};

/// Numerical convention used by checkpoint-wide NVFP4 scales.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NvFp4ScaleMode {
    /// Stored values multiply quantized inputs and weights directly.
    Multiplier,
    /// Stored values divide quantized inputs and weights.
    Divisor,
}

impl NvFp4ScaleMode {
    pub(super) fn multiplier(self, value: f32) -> Result<f32> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::InvalidNvFp4("global scale must be finite and positive"));
        }
        Ok(match self {
            Self::Multiplier => value,
            Self::Divisor => value.recip(),
        })
    }

    pub(crate) fn from_names(weight: &str, input: &str) -> Result<Self> {
        match (
            weight.ends_with(".weight_scale_2"),
            input.ends_with(".input_scale"),
            weight.ends_with(".weight_global_scale"),
            input.ends_with(".input_global_scale"),
        ) {
            (true, true, false, false) => Ok(Self::Multiplier),
            (false, false, true, true) => Ok(Self::Divisor),
            _ => Err(Error::InvalidNvFp4("mixed or unknown global scale convention")),
        }
    }
}
