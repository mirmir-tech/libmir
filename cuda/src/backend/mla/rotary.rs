//! Rotary frequencies of the latent attention's rotary part, computed once on
//! the host; the kernels read them per pair.

use models::semantic::{PositionEncodingSpec, RopeScalingSpec};

use crate::{Error, Result};

/// Inverse frequencies of the `rope_dim / 2` pairs and the factor on their
/// cosine and sine (`YaRN`'s attention factor, `mscale / mscale_all_dim` in
/// `DeepSeek` checkpoints).
// Rotary widths and context lengths are far below `f64`'s exact integers.
#[allow(clippy::cast_precision_loss)]
pub(super) fn frequencies(
    position: &PositionEncodingSpec,
    rope_dim: usize,
) -> Result<(Vec<f32>, f32)> {
    let PositionEncodingSpec::Rotary(rotary) = position else {
        return Err(unsupported("latent attention without a rotary part"));
    };
    let dim = rope_dim as f64;
    let base = |pair: usize| rotary.theta.powf(2.0 * pair as f64 / dim);
    let pairs = 0..rope_dim / 2;
    let (inverse, scale): (Vec<f64>, f64) = match rotary.scaling {
        None => (pairs.map(|pair| base(pair).recip()).collect(), 1.0),
        Some(RopeScalingSpec::Linear { factor }) => {
            (pairs.map(|pair| (factor * base(pair)).recip()).collect(), 1.0)
        },
        Some(RopeScalingSpec::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_context_len,
            attention_factor,
            truncate,
        }) => {
            // Pairs rotating faster than `beta_fast` turns over the original
            // context keep their frequency, slower than `beta_slow` ones are
            // interpolated by `factor`, with a linear ramp between.
            let original = original_context_len as f64;
            let correction = |turns: f64| {
                dim * (original / (turns * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * rotary.theta.ln())
            };
            let (mut low, mut high) = (correction(beta_fast), correction(beta_slow));
            if truncate {
                (low, high) = (low.floor(), high.ceil());
            }
            let low = low.max(0.0);
            let mut high = high.min(dim - 1.0);
            if (high - low).abs() < f64::EPSILON {
                high += 0.001;
            }
            let inverse = pairs
                .map(|pair| {
                    let ramp = ((pair as f64 - low) / (high - low)).clamp(0.0, 1.0);
                    let frequency = base(pair).recip();
                    (frequency / factor).mul_add(ramp, frequency * (1.0 - ramp))
                })
                .collect();
            (inverse, attention_factor)
        },
        Some(RopeScalingSpec::PiecewiseFrequency { .. }) => {
            return Err(unsupported("latent attention with piecewise rotary frequencies"));
        },
    };
    #[allow(clippy::cast_possible_truncation)]
    Ok((inverse.into_iter().map(|value| value as f32).collect(), scale as f32))
}

fn unsupported(what: &str) -> Error {
    Error::UnsupportedDecoderLayer(format!("CUDA has no {what}"))
}

#[cfg(test)]
mod tests {
    use models::semantic::{RotaryLayoutSpec, RotarySpec};

    use super::*;

    fn rotary(scaling: Option<RopeScalingSpec>) -> PositionEncodingSpec {
        PositionEncodingSpec::Rotary(RotarySpec {
            theta: 10_000.0,
            partial_factor: 1.0,
            layout: RotaryLayoutSpec::Standard,
            algorithm: None,
            scaling,
        })
    }

    /// `DeepSeek-V2-Lite`'s `YaRN`: the first pairs keep their frequency, the
    /// last are divided by the factor.
    #[test]
    fn interpolates_slow_pairs_only() -> Result<()> {
        let (plain, _) = frequencies(&rotary(None), 64)?;
        let (yarn, scale) = frequencies(
            &rotary(Some(RopeScalingSpec::Yarn {
                factor: 40.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_context_len: 4096,
                attention_factor: 1.0,
                truncate: true,
            })),
            64,
        )?;
        assert!((scale - 1.0).abs() < f32::EPSILON);
        assert!((yarn[0] - plain[0]).abs() < f32::EPSILON);
        assert!((yarn[31] - plain[31] / 40.0).abs() <= plain[31] * 1e-6);
        assert!(yarn[15] < plain[15] && yarn[15] > plain[15] / 40.0);
        Ok(())
    }
}
