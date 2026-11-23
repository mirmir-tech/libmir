//! Rotary inverse frequencies computed on the host, for backends that rotate
//! with explicit frequencies (latent attention's rotary part).

use super::{PositionEncodingSpec, RopeScalingSpec, RotarySpec};
use crate::error::{ModelsError, Result};

impl RotarySpec {
    /// The leading dimensions of a `head_dim`-wide head that rotate; a
    /// partial factor must give a whole, even count.
    // Head widths are far below `f64`'s exact integers.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn rotated_dimensions(&self, head_dim: usize) -> Result<usize> {
        let dimensions = head_dim as f64 * self.partial_factor;
        if !dimensions.is_finite() || dimensions <= 0.0 || dimensions.fract() != 0.0 {
            return Err(ModelsError::InvalidConfig(
                "partial RoPE does not give a whole number of dimensions".into(),
            ));
        }
        let dimensions = dimensions as usize;
        if dimensions > head_dim || !dimensions.is_multiple_of(2) {
            return Err(ModelsError::InvalidConfig(
                "partial RoPE must rotate an even number of the head's dimensions".into(),
            ));
        }
        Ok(dimensions)
    }

    /// The width the pair frequencies spread over: the whole head for Gemma
    /// 4's `proportional` `RoPE`, which rotates only its leading pairs, else
    /// the rotated dimensions.
    pub fn pairing_dimensions(&self, head_dim: usize) -> Result<usize> {
        match self.algorithm.as_deref() {
            Some("proportional") => Ok(head_dim),
            None | Some("default") => self.rotated_dimensions(head_dim),
            Some(kind) => Err(ModelsError::InvalidConfig(format!("unsupported RoPE type {kind}"))),
        }
    }
}

impl PositionEncodingSpec {
    /// Inverse frequencies of the `dim / 2` rotary pairs and the factor on
    /// their cosine and sine (`YaRN`'s attention factor, `mscale /
    /// mscale_all_dim` when the config declares both); `None` without a
    /// rotary encoding.
    // Rotary widths and context lengths are far below `f64`'s exact integers.
    #[allow(clippy::cast_precision_loss)]
    pub fn inverse_frequencies(&self, dim: usize) -> Result<Option<(Vec<f64>, f64)>> {
        let Self::Rotary(rotary) = self else {
            return Ok(None);
        };
        let width = dim as f64;
        let base = |pair: usize| rotary.theta.powf(2.0 * pair as f64 / width);
        let pairs = 0..dim / 2;
        Ok(Some(match rotary.scaling {
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
                // Pairs rotating faster than `beta_fast` turns over the
                // original context keep their frequency, slower than
                // `beta_slow` ones are interpolated by `factor`, with a
                // linear ramp between.
                let original = original_context_len as f64;
                let correction = |turns: f64| {
                    width * (original / (turns * 2.0 * std::f64::consts::PI)).ln()
                        / (2.0 * rotary.theta.ln())
                };
                let (mut low, mut high) = (correction(beta_fast), correction(beta_slow));
                if truncate {
                    (low, high) = (low.floor(), high.ceil());
                }
                let low = low.max(0.0);
                let mut high = high.min(width - 1.0);
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
            Some(RopeScalingSpec::LongRope { factors, attention_factor, .. }) => {
                let factors = factors.as_slice();
                if factors.len() != dim / 2 {
                    return Err(ModelsError::InvalidConfig(format!(
                        "LongRoPE declares {} factors for {} rotary pairs",
                        factors.len(),
                        dim / 2
                    )));
                }
                let inverse = pairs
                    .zip(factors)
                    .map(|(pair, &factor)| (f64::from(factor) * base(pair)).recip());
                (inverse.collect(), attention_factor)
            },
            Some(RopeScalingSpec::PiecewiseFrequency {
                factor,
                low_frequency_factor: low,
                high_frequency_factor: high,
                original_context_len,
            }) => (
                pairs
                    .map(|pair| {
                        llama3(base(pair).recip(), factor, (low, high), original_context_len)
                    })
                    .collect(),
                1.0,
            ),
        }))
    }
}

/// transformers' `_compute_llama3_parameters` for one pair: wavelengths
/// beyond `original / low` divide the frequency by `factor`, those within
/// `original / high` keep it, and those between blend. Equal factors
/// (Llama 4) leave no span to blend, so the stop is hard.
// Context lengths are far below `f64`'s exact integers.
#[allow(clippy::cast_precision_loss)]
fn llama3(frequency: f64, factor: f64, (low, high): (f64, f64), original: usize) -> f64 {
    let original = original as f64;
    let wavelength = std::f64::consts::TAU / frequency;
    if wavelength > original / low {
        frequency / factor
    } else if wavelength < original / high || high <= low {
        frequency
    } else {
        let smooth = (original / wavelength - low) / (high - low);
        smooth.mul_add(frequency, (1.0 - smooth) * frequency / factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::{RotaryLayoutSpec, RotarySpec};

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
        let (plain, _) = rotary(None)
            .inverse_frequencies(64)?
            .ok_or(ModelsError::InvalidConfig("plain".into()))?;
        let (yarn, scale) = rotary(Some(RopeScalingSpec::Yarn {
            factor: 40.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_context_len: 4096,
            attention_factor: 1.0,
            truncate: true,
        }))
        .inverse_frequencies(64)?
        .ok_or(ModelsError::InvalidConfig("yarn".into()))?;
        assert!((scale - 1.0).abs() < f64::EPSILON);
        assert!((yarn[0] - plain[0]).abs() < f64::EPSILON);
        assert!((yarn[31] - plain[31] / 40.0).abs() <= plain[31] * 1e-9);
        assert!(yarn[15] < plain[15] && yarn[15] > plain[15] / 40.0);
        assert_eq!(PositionEncodingSpec::None.inverse_frequencies(64)?, None);
        Ok(())
    }

    /// Llama 3.1 blends the middle pairs; Llama 4's equal factors stop hard at
    /// the original context: with theta 500000 and 128 dimensions its pairs'
    /// wavelengths 6694 and 8222 straddle 8192.
    #[test]
    fn llama3_scaling_blends_and_llama4_stops_hard() -> Result<()> {
        let llama = |low, high| {
            PositionEncodingSpec::Rotary(RotarySpec {
                theta: 500_000.0,
                ..match rotary(Some(RopeScalingSpec::PiecewiseFrequency {
                    factor: 8.0,
                    low_frequency_factor: low,
                    high_frequency_factor: high,
                    original_context_len: 8192,
                })) {
                    PositionEncodingSpec::Rotary(rotary) => rotary,
                    PositionEncodingSpec::None => unreachable!("built rotary"),
                }
            })
            .inverse_frequencies(128)
        };
        let plain = |pair: u32| 500_000.0_f64.powf(-2.0 * f64::from(pair) / 128.0);
        let (blended, _) = llama(1.0, 4.0)?.ok_or(ModelsError::InvalidConfig("3.1".into()))?;
        let (hard, _) = llama(1.0, 1.0)?.ok_or(ModelsError::InvalidConfig("4".into()))?;
        assert!((blended[0] - plain(0)).abs() < f64::EPSILON);
        assert!((blended[63] - plain(63) / 8.0).abs() <= plain(63) * 1e-12);
        assert!(blended[30] < plain(30) && blended[30] > plain(30) / 8.0);
        for (pair, value) in (0_u32..).zip(&hard) {
            let wavelength = std::f64::consts::TAU / plain(pair);
            let expected = if wavelength > 8192.0 {
                plain(pair) / 8.0
            } else {
                plain(pair)
            };
            assert!((value - expected).abs() <= expected * 1e-12, "pair {pair}");
        }
        Ok(())
    }
}
