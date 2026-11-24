//! Scaled `RoPE` inverse frequencies against independent references.

use super::super::{PositionEncodingSpec, RopeScalingSpec, RotaryLayoutSpec, RotarySpec};
use crate::error::Result;

fn rotary(theta: f64, scaling: RopeScalingSpec) -> PositionEncodingSpec {
    PositionEncodingSpec::Rotary(RotarySpec {
        theta,
        partial_factor: 1.0,
        layout: RotaryLayoutSpec::Standard,
        algorithm: crate::semantic::RotaryAlgorithm::Default,
        scaling: Some(scaling),
    })
}

fn periods(position: &PositionEncodingSpec, dim: usize) -> Result<Vec<f64>> {
    let (inverse, _) = position.inverse_frequencies(dim)?.unwrap_or_default();
    Ok(inverse.iter().map(|frequency| frequency.recip()).collect())
}

/// Llama 3: fast pairs keep their period, slow ones are slowed by `factor`.
#[test]
fn piecewise_frequency_keeps_fast_and_slows_slow_pairs() -> Result<()> {
    let position = rotary(
        10_000.0,
        RopeScalingSpec::PiecewiseFrequency {
            factor: 8.0,
            low_frequency_factor: 1.0,
            high_frequency_factor: 4.0,
            original_context_len: 8192,
        },
    );
    let periods = periods(&position, 8)?;
    assert_eq!(periods.len(), 4);
    for (actual, expected) in periods.iter().zip([1.0, 10.0, 100.0]) {
        assert!((actual - expected).abs() < expected * 1e-12);
    }
    assert!(periods[3] > 4_000.0 && periods[3] < 5_000.0);
    Ok(())
}

/// GPT-OSS's continuous `YaRN`, checked against an independent calculation
/// with no floor or ceiling on the correction range.
#[test]
fn continuous_yarn_matches_gpt_oss_reference() -> Result<()> {
    let yarn = |truncate| {
        rotary(
            150_000.0,
            RopeScalingSpec::Yarn {
                factor: 32.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_context_len: 4096,
                attention_factor: 1.0,
                truncate,
            },
        )
    };
    let expected = [
        1_f64, 1.451_285_49, 2.106_229_56, 3.056_740_39, 4.436_202_96, 6.438_196_97, 9.343_661_81,
        13.560_320_8, 19.679_896_7, 31.540_073_9, 51.719_676_3, 86.266_024_1, 147.167_912,
        259.043_244, 477.602_273, 950.026_604, 2_190.657_67, 7_732.833_61, 26_103.654_4,
        37_883.854_8, 54_980.288_6, 79_792.094_8, 115_801.109, 168_060.469, 243_903.719,
        353_973.927, 513_717.223, 745_550.35, 1_082_006.4, 1_570_300.18, 2_278_953.87,
        3_307_412.67,
    ];
    let continuous = periods(&yarn(false), 64)?;
    assert_eq!(continuous.len(), expected.len());
    for (actual, expected) in continuous.iter().zip(expected) {
        assert!((actual - expected).abs() <= expected.max(1.0) * 3.0e-6);
    }
    let truncated = periods(&yarn(true), 64)?;
    assert!((continuous[12] - truncated[12]).abs() > continuous[12] * 0.01);
    Ok(())
}

/// Sidecars written before the algorithm was typed omit it.
#[test]
fn rotary_specs_without_an_algorithm_rotate_their_dimensions() -> Result<()> {
    let spec: RotarySpec =
        toml::from_str("theta = 10000.0\npartial_factor = 1.0\n[layout]\nkind = \"standard\"\n")?;
    assert_eq!(spec.algorithm, super::super::RotaryAlgorithm::Default);
    assert!(!toml::to_string(&spec)?.contains("algorithm"));
    Ok(())
}
