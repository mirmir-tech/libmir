//! Declared per-pair rotary periods (Phi-3 `LongRoPE`).

use mircuda::bf16;
use models::layout::RopeFactors;

use super::{Runtime, batched, fused, test_values};
use crate::{
    Result,
    kernels::{
        QkvNormalization, QkvPostprocessSpec, RotaryEmbedding, RotaryPeriods, RotaryScaling,
    },
};

const FACTORS: [f64; 4] = [1.0, 1.5, 3.25, 7.0];
const ATTENTION_FACTOR: f32 = 1.1902;

fn spec() -> Result<QkvPostprocessSpec> {
    let factors = RopeFactors::new(&FACTORS)?;
    Ok(QkvPostprocessSpec {
        tokens: 2,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 8,
        value_head_dim: 8,
        rotary: Some(RotaryEmbedding {
            dimensions: 8,
            pairing: 8,
            theta: 10_000.0,
            scaling: RotaryScaling::periods(RotaryPeriods {
                factors,
                attention_factor: ATTENTION_FACTOR,
            }),
        }),
        epsilon: 1.0e-6,
        normalization: QkvNormalization::NONE,
        temperature: None,
    })
}

#[test]
fn periods_scale_each_pair_and_its_magnitude() -> Result<()> {
    let runtime = Runtime::new()?;
    let spec = spec()?;
    let (packed, weight) = (test_values(48, 0)?, test_values(8, 0)?);
    let (query, key, value) = fused(&runtime, spec, &packed, &weight, &weight, 1000)?;
    for (token, position) in [(0, 1000), (1, 1001)] {
        let row = &packed[token * 24..token * 24 + 24];
        let at = token * 8..token * 8 + 8;
        assert_rotated(&query[at.clone()], &row[..8], position);
        assert_rotated(&key[at.clone()], &row[8..16], position);
        assert_eq!(&value[at], &row[16..]);
    }
    let fused_at_three = fused(&runtime, spec, &packed, &weight, &weight, 3)?;
    assert_eq!(batched(&runtime, spec, &packed, &weight, &weight)?, fused_at_three);
    Ok(())
}

/// The kernel against `f64` rotation of `input` at `position` with each
/// pair's period multiplied by its factor, to one BF16 rounding.
fn assert_rotated(actual: &[bf16], input: &[bf16], position: u16) {
    let input: Vec<f64> = input.iter().map(|value| f64::from(value.to_f32())).collect();
    for (index, factor) in (0_u8..).zip(FACTORS) {
        let pair = usize::from(index);
        let period = 10_000.0_f64.powf(2.0 * f64::from(index) / 8.0);
        let angle = f64::from(position) / (period * factor);
        let (sine, cosine) = angle.sin_cos();
        let magnitude = f64::from(ATTENTION_FACTOR);
        let (first, second) = (input[pair], input[pair + 4]);
        let expected = [
            first.mul_add(cosine, -second * sine) * magnitude,
            first.mul_add(sine, second * cosine) * magnitude,
        ];
        for (actual, expected) in [actual[pair], actual[pair + 4]].into_iter().zip(expected) {
            let error = (f64::from(actual.to_f32()) - expected).abs();
            assert!(
                error <= expected.abs().max(1.0) / 128.0,
                "pair {pair} at {position}: {actual} vs {expected}"
            );
        }
    }
}
