//! Rotary scaling from the semantic spec against its own frequencies.

use models::semantic::{
    PositionEncodingSpec, RopeScalingSpec, RotaryAlgorithm, RotaryLayoutSpec, RotarySpec,
};

use super::scaling;
use crate::Result;

/// Qwen3's long-context `YaRN` (factor 4 over 32768 positions, with its
/// `0.1 ln(factor) + 1` attention factor): the kernel's base frequency over
/// each pair's period factor is the spec's `YaRN` frequency, and the cosine
/// and sine take the spec's factor.
#[test]
fn yarn_scaling_is_the_specs_frequencies_and_factor() -> Result<()> {
    let attention_factor = 0.1f64.mul_add(4.0_f64.ln(), 1.0);
    let position = PositionEncodingSpec::Rotary(RotarySpec {
        theta: 1_000_000.0,
        partial_factor: 1.0,
        layout: RotaryLayoutSpec::Standard,
        algorithm: RotaryAlgorithm::Default,
        scaling: Some(RopeScalingSpec::Yarn {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_context_len: 32_768,
            attention_factor,
            truncate: true,
        }),
    });
    let (expected, _) = position.inverse_frequencies(128)?.unwrap_or_default();
    let periods = scaling(&position, (128, 128))?
        .periods
        .ok_or(crate::Error::InvalidDecoderKernel("YaRN has per-pair periods"))?;
    assert!((f64::from(periods.attention_factor) - attention_factor).abs() < 1e-6);
    let factors = periods.factors.as_slice();
    assert_eq!(factors.len(), 64);
    for (pair, (&factor, expected)) in (0_u32..).zip(factors.iter().zip(expected)) {
        let base = 1_000_000.0_f64.powf(-2.0 * f64::from(pair) / 128.0);
        let actual = base / f64::from(factor);
        assert!(
            (actual - expected).abs() <= 1e-6 * expected,
            "pair {pair}: {actual} vs {expected}"
        );
    }
    // The first pairs keep their frequency, the last are interpolated.
    assert!((factors[0] - 1.0).abs() < 1e-6 && (factors[63] - 4.0).abs() < 1e-4);
    Ok(())
}
