use models::semantic::{AttentionSpec, PositionEncodingSpec, RopeScalingSpec};

use super::config::{to_f32, unsupported};
use crate::{Result, kernels::RotaryScaling};

/// The layer's long-context frequency scaling; `YaRN` has its own attention
/// factor and runtime (`clamped_routed`).
pub(super) fn scaling(spec: &AttentionSpec) -> Result<RotaryScaling> {
    let PositionEncodingSpec::Rotary(rotary) = &spec.position else {
        return Ok(RotaryScaling::NONE);
    };
    match rotary.scaling {
        None => Ok(RotaryScaling::NONE),
        Some(RopeScalingSpec::Linear { factor }) => {
            Ok(RotaryScaling::linear(to_f32("RoPE factor", factor)?))
        },
        Some(RopeScalingSpec::PiecewiseFrequency {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_context_len,
        }) => Ok(RotaryScaling::piecewise(
            to_f32("RoPE factor", factor)?,
            to_f32("RoPE low-frequency factor", low_frequency_factor)?,
            to_f32("RoPE high-frequency factor", high_frequency_factor)?,
            to_f32("RoPE original context", f64::from(u32::try_from(original_context_len)?))?,
        )),
        Some(RopeScalingSpec::Yarn { .. }) => {
            Err(unsupported("dense CUDA attention has no YaRN rotary scaling"))
        },
    }
}
