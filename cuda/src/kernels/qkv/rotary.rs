//! Rotary position embedding applied by the QKV postprocess kernels.

use super::super::geometry::narrow;
use crate::Result;

/// Rotary embedding over the leading `dimensions` of each query and key head.
/// Frequencies are spread over `pairing` dimensions; dimensions beyond
/// `dimensions` pass through unrotated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryEmbedding {
    pub dimensions: usize,
    pub pairing: usize,
    pub theta: f32,
    pub scaling: RotaryScaling,
}

/// Long-context frequency scaling.
///
/// Each pair's frequency `f` becomes `f * ((1 - s) / factor + s)` with
/// `s = clamp(slope * f - offset, 0, 1)`, so `factor` 1 leaves it unchanged,
/// `slope` and `offset` 0 scale linearly, and Llama 3's piecewise scaling
/// blends between the two bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryScaling {
    pub factor: f32,
    pub slope: f32,
    pub offset: f32,
}

impl RotaryScaling {
    pub const NONE: Self = Self { factor: 1.0, slope: 0.0, offset: 0.0 };

    #[must_use]
    pub const fn linear(factor: f32) -> Self {
        Self { factor, slope: 0.0, offset: 0.0 }
    }

    /// Llama 3: pairs with wavelengths beyond `original / low` take the full
    /// factor, those within `original / high` none, and those between blend.
    #[must_use]
    pub fn piecewise(factor: f32, low: f32, high: f32, original: f32) -> Self {
        let span = high - low;
        Self {
            factor,
            slope: original / (std::f32::consts::TAU * span),
            offset: low / span,
        }
    }

    fn valid(self) -> bool {
        self.factor.is_finite()
            && self.factor > 0.0
            && self.slope.is_finite()
            && self.offset.is_finite()
    }
}

/// The rotary kernel arguments; heads without a rotary embedding rotate zero
/// dimensions and pass through.
#[derive(Clone, Copy, Debug)]
pub struct RotaryArguments {
    pub rotary_dim: u32,
    pub pairing_dim: u32,
    pub theta: f32,
    pub scaling: RotaryScaling,
}

impl RotaryEmbedding {
    pub(crate) fn fits(self, head_dim: usize) -> bool {
        self.dimensions != 0
            && self.dimensions <= self.pairing
            && self.pairing <= head_dim
            && self.dimensions.is_multiple_of(2)
            && self.pairing.is_multiple_of(2)
            && self.theta.is_finite()
            && self.theta > 0.0
            && self.scaling.valid()
    }

    pub fn arguments(rotary: Option<Self>) -> Result<RotaryArguments> {
        rotary.map_or(
            Ok(RotaryArguments {
                rotary_dim: 0,
                pairing_dim: 0,
                theta: 1.0,
                scaling: RotaryScaling::NONE,
            }),
            |rotary| {
                Ok(RotaryArguments {
                    rotary_dim: narrow(rotary.dimensions)?,
                    pairing_dim: narrow(rotary.pairing)?,
                    theta: rotary.theta,
                    scaling: rotary.scaling,
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::RotaryScaling;

    /// The kernel's blend against `transformers`' Llama 3 frequencies
    /// (`_compute_llama3_parameters`) for Llama 3.1's parameters.
    #[test]
    fn piecewise_scaling_matches_llama3_frequencies() {
        let (factor, low, high, original) = (8.0_f64, 1.0, 4.0, 8192.0);
        let scaling = RotaryScaling::piecewise(8.0, 1.0, 4.0, 8192.0);
        for pair in 0..64_u32 {
            let frequency = 500_000.0_f64.powf(-2.0 * f64::from(pair) / 128.0);
            let wavelength = std::f64::consts::TAU / frequency;
            let expected = if wavelength < original / high {
                frequency
            } else if wavelength > original / low {
                frequency / factor
            } else {
                let smooth = (original / wavelength - low) / (high - low);
                (1.0 - smooth) * frequency / factor + smooth * frequency
            };
            let smooth = f64::from(scaling.slope)
                .mul_add(frequency, -f64::from(scaling.offset))
                .clamp(0.0, 1.0);
            let actual = frequency * ((1.0 - smooth) / f64::from(scaling.factor) + smooth);
            assert!(
                (actual - expected).abs() <= 1e-5 * expected,
                "pair {pair}: {actual} vs {expected}"
            );
        }
    }
}
