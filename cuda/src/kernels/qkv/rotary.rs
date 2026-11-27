//! Rotary position embedding applied by the QKV postprocess kernels.

use mircuda::CompileOptions;
use models::layout::RopeFactors;

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
/// blends between the two bounds. [`RotaryPeriods`] apply on top of the
/// blend, which `LongRoPE` leaves at identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryScaling {
    pub factor: f32,
    pub slope: f32,
    pub offset: f32,
    pub periods: Option<RotaryPeriods>,
}

/// Declared per-pair period factors (Phi-3 `LongRoPE`).
///
/// Pair `i` rotates at its frequency divided by `factors[i]`, and its cosine
/// and sine scale by `attention_factor`. The kernel variant compiles both in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryPeriods {
    pub factors: RopeFactors,
    pub attention_factor: f32,
}

impl RotaryPeriods {
    /// Defines the factor table and attention factor the postprocess kernels
    /// compile in, as `f32` literals that round-trip exactly.
    pub(super) fn define_macros(self, options: &mut CompileOptions) {
        let factors = self.factors.as_slice().iter().map(|factor| format!("{factor:e}f"));
        let factors = factors.collect::<Vec<_>>().join(",");
        options.extra_options.extend([
            format!("--define-macro=LIBMIR_ROPE_PAIR_FACTORS={factors}"),
            format!("--define-macro=LIBMIR_ROPE_ATTENTION_FACTOR={:e}f", self.attention_factor),
        ]);
    }
}

impl RotaryScaling {
    pub const NONE: Self = Self {
        factor: 1.0,
        slope: 0.0,
        offset: 0.0,
        periods: None,
    };

    #[must_use]
    pub const fn linear(factor: f32) -> Self {
        Self {
            factor,
            slope: 0.0,
            offset: 0.0,
            periods: None,
        }
    }

    #[must_use]
    pub const fn periods(periods: RotaryPeriods) -> Self {
        Self { periods: Some(periods), ..Self::NONE }
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
            periods: None,
        }
    }

    fn valid(self, rotary_pairs: usize) -> bool {
        self.factor.is_finite()
            && self.factor > 0.0
            && self.slope.is_finite()
            && self.offset.is_finite()
            && self.periods.is_none_or(|periods| {
                periods.factors.as_slice().len() == rotary_pairs
                    && periods.attention_factor.is_finite()
                    && periods.attention_factor > 0.0
            })
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
            && self.scaling.valid(self.dimensions / 2)
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
    use models::semantic::{PositionEncodingSpec, RopeScalingSpec, RotaryLayoutSpec, RotarySpec};

    use super::RotaryScaling;

    /// The kernel's blend against the semantic spec's Llama 3 frequencies
    /// (transformers' `_compute_llama3_parameters`) for Llama 3.1.
    #[test]
    fn piecewise_scaling_matches_llama3_frequencies() -> models::Result<()> {
        let scaling = RotaryScaling::piecewise(8.0, 1.0, 4.0, 8192.0);
        let position = PositionEncodingSpec::Rotary(RotarySpec {
            theta: 500_000.0,
            partial_factor: 1.0,
            layout: RotaryLayoutSpec::Standard,
            algorithm: models::semantic::RotaryAlgorithm::Default,
            scaling: Some(RopeScalingSpec::PiecewiseFrequency {
                factor: 8.0,
                low_frequency_factor: 1.0,
                high_frequency_factor: 4.0,
                original_context_len: 8192,
            }),
        });
        let (expected, _) = position.inverse_frequencies(128)?.unwrap_or_default();
        assert_eq!(expected.len(), 64);
        for (pair, expected) in (0_u32..).zip(expected) {
            let frequency = 500_000.0_f64.powf(-2.0 * f64::from(pair) / 128.0);
            let smooth = f64::from(scaling.slope)
                .mul_add(frequency, -f64::from(scaling.offset))
                .clamp(0.0, 1.0);
            let actual = frequency * ((1.0 - smooth) / f64::from(scaling.factor) + smooth);
            assert!(
                (actual - expected).abs() <= 1e-5 * expected,
                "pair {pair}: {actual} vs {expected}"
            );
        }
        Ok(())
    }
}
