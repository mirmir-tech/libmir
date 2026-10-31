use mircuda::bf16;
use runtime::backend::SamplingLogits;

use super::{super::*, copy, sample};
use crate::{CudaConfig, Result};

fn policy(vocab_size: usize, top_p: f32, temperature: f32, draw: f32) -> SamplingLogits {
    SamplingLogits::Sample {
        vocab_size,
        temperature,
        top_p,
        top_k: 0,
        draw,
    }
}

#[test]
fn unbounded_nucleus_keeps_the_smallest_prefix_reaching_top_p() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let logits = copy(&backend, &[3.0, 2.0, 1.0, 0.0])?;
    let mut sampler = backend.prepare_device_sampler_bf16(4)?;
    // Probabilities are about 0.64, 0.24, 0.09 and 0.03.
    for (top_p, draw, expected) in
        [(0.6, 0.99, 0), (0.85, 0.0, 0), (0.85, 0.99, 1), (0.95, 0.99, 2)]
    {
        assert_eq!(sample(&backend, &mut sampler, &logits, policy(4, top_p, 1.0, draw))?, expected);
    }
    Ok(())
}

#[test]
fn unbounded_nucleus_matches_the_host_rule_over_a_large_vocabulary() -> Result<()> {
    const VOCAB: usize = 131_072;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let values = (0..VOCAB)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
            bf16::from_f32(unit * 24.0 - 12.0).to_f32()
        })
        .collect::<Vec<_>>();
    let logits = copy(&backend, &values)?;
    let mut sampler = backend.prepare_device_sampler_bf16(VOCAB)?;
    for (top_p, temperature) in [(0.95, 1.0), (0.5, 0.7), (0.99, 1.3)] {
        for draw in [0.0, 0.13, 0.5, 0.77, 0.999] {
            let device =
                sample(&backend, &mut sampler, &logits, policy(VOCAB, top_p, temperature, draw))?;
            assert_eq!(
                device,
                reference(&values, top_p, temperature, draw),
                "{top_p} {temperature} {draw}"
            );
            let again =
                sample(&backend, &mut sampler, &logits, policy(VOCAB, top_p, temperature, draw))?;
            assert_eq!(device, again);
        }
    }
    Ok(())
}

/// The host sampler's nucleus: descending scores until the kept mass reaches
/// `top_p`, then the draw walks the kept tokens in vocabulary order.
fn reference(values: &[f32], top_p: f32, temperature: f32, draw: f32) -> u32 {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weight = |score: f32| (f64::from(score - maximum) / f64::from(temperature)).exp();
    let total = values.iter().map(|score| weight(*score)).sum::<f64>();
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
    let mut kept = 0.0;
    let mut threshold = f32::NEG_INFINITY;
    for index in order {
        kept += weight(values[index]) / total;
        threshold = values[index];
        if kept >= f64::from(top_p) {
            break;
        }
    }
    let mass = values
        .iter()
        .filter(|score| **score >= threshold)
        .map(|score| weight(*score))
        .sum::<f64>();
    let target = mass * f64::from(draw);
    let mut cumulative = 0.0;
    for (index, score) in values.iter().enumerate() {
        if *score >= threshold {
            cumulative += weight(*score);
            if cumulative >= target && weight(*score) > 0.0 {
                return u32::try_from(index).unwrap_or(u32::MAX);
            }
        }
    }
    0
}
