use models::semantic::{AttentionSpec, PositionEncodingSpec, RopeScalingSpec};

use super::config::{to_f32, to_f64, unsupported};
use crate::{
    Result,
    kernels::{RotaryEmbedding, RotaryPeriods, RotaryScaling},
};

/// The layer's rotary embedding from its semantic spec; `None` for layers
/// without position encoding (Cohere 2's full-attention layers).
pub(super) fn embedding(spec: &AttentionSpec, head_dim: usize) -> Result<Option<RotaryEmbedding>> {
    let PositionEncodingSpec::Rotary(rotary) = &spec.position else {
        return Ok(None);
    };
    let dimensions = rotary.rotated_dimensions(head_dim)?;
    let pairing = rotary.pairing_dimensions(head_dim)?;
    Ok(Some(RotaryEmbedding {
        dimensions,
        pairing,
        theta: to_f32("RoPE theta", rotary.theta)?,
        scaling: scaling(spec, (dimensions, pairing))?,
    }))
}

/// The layer's long-context frequency scaling; `YaRN` has its own attention
/// factor and runtime (`clamped_routed`).
fn scaling(spec: &AttentionSpec, (dimensions, pairing): (usize, usize)) -> Result<RotaryScaling> {
    let PositionEncodingSpec::Rotary(rotary) = &spec.position else {
        return Ok(RotaryScaling::NONE);
    };
    match rotary.scaling {
        None => Ok(RotaryScaling::NONE),
        Some(RopeScalingSpec::Linear { factor }) => {
            Ok(RotaryScaling::linear(to_f32("RoPE factor", factor)?))
        },
        Some(RopeScalingSpec::PiecewiseFrequency {
            low_frequency_factor,
            high_frequency_factor,
            ..
        }) if high_frequency_factor <= low_frequency_factor => {
            pair_periods(spec, (dimensions, pairing))
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
        Some(RopeScalingSpec::LongRope { factors, attention_factor, .. }) => {
            Ok(RotaryScaling::periods(RotaryPeriods {
                factors,
                attention_factor: to_f32("LongRoPE attention factor", attention_factor)?,
            }))
        },
    }
}

/// Llama 4's hard frequency stop as per-pair period factors: each rotated
/// pair's base frequency over the semantic spec's scaled one, so the
/// kernels divide by the factors the spec defines.
fn pair_periods(
    spec: &AttentionSpec,
    (dimensions, pairing): (usize, usize),
) -> Result<RotaryScaling> {
    let PositionEncodingSpec::Rotary(rotary) = &spec.position else {
        return Ok(RotaryScaling::NONE);
    };
    let Some((scaled, _)) = spec.position.inverse_frequencies(pairing)? else {
        return Ok(RotaryScaling::NONE);
    };
    let width = to_f64(pairing)?;
    let factors = (0..dimensions / 2)
        .zip(&scaled)
        .map(|(pair, scaled)| Ok(rotary.theta.powf(-2.0 * to_f64(pair)? / width) / scaled))
        .collect::<Result<Vec<_>>>()?;
    Ok(RotaryScaling::periods(RotaryPeriods {
        factors: models::layout::RopeFactors::new(&factors)?,
        attention_factor: 1.0,
    }))
}
