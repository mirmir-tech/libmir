//! Exact conversions between the integer geometry and the floating-point
//! rules of the processor.

use crate::error::Result;

/// A count as a float, exactly.
pub(super) fn real(value: usize) -> Result<f64> {
    Ok(f64::from(u32::try_from(value)?))
}

/// A float the rule has already floored or ceiled, as a count.
pub(super) fn whole(value: f64) -> Result<usize> {
    Ok(value.to_string().parse()?)
}

/// A configuration float in the single precision the pixels use.
pub(super) fn narrow(value: f64) -> Result<f32> {
    Ok(value.to_string().parse()?)
}
