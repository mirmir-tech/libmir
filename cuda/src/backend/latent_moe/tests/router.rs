use mircuda::bf16;

use super::{copy, read};
use crate::{
    CudaConfig, Result,
    backend::CudaBackend,
    kernels::{ReluSquaredBf16, SigmoidRouter, SigmoidRouterOutputs, SigmoidRouterSpec},
};

const SPEC: SigmoidRouterSpec = SigmoidRouterSpec {
    hidden: 256,
    experts: 64,
    top_k: 6,
    routed_scale: 2.5,
};

/// Deterministic values in `[-scale, scale]`.
fn pattern(elements: usize, seed: u64, scale: f32) -> Vec<bf16> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..elements)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit =
                f32::from(u16::try_from(state >> 48).unwrap_or_default()) / f32::from(u16::MAX);
            bf16::from_f32(unit.mul_add(2.0, -1.0) * scale)
        })
        .collect()
}

/// Host routing of one token: logits in FP64, selection by sigmoid + bias
/// with the lower index winning ties, renormalized and scaled weights.
fn reference(
    input: &[bf16],
    weight: &[bf16],
    bias: &[f32],
    token: usize,
) -> (Vec<u32>, Vec<f64>, Vec<f64>) {
    let hidden = SPEC.hidden;
    let logits = (0..SPEC.experts)
        .map(|expert| {
            (0..hidden).fold(0.0, |sum, column| {
                f64::from(input[token * hidden + column].to_f32())
                    .mul_add(f64::from(weight[expert * hidden + column].to_f32()), sum)
            })
        })
        .collect::<Vec<f64>>();
    let scores = logits.iter().map(|logit| 1.0 / (1.0 + (-logit).exp())).collect::<Vec<_>>();
    let mut order = (0..SPEC.experts).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        let choice = |expert: &usize| scores[*expert] + f64::from(bias[*expert]);
        choice(right).total_cmp(&choice(left)).then(left.cmp(right))
    });
    let chosen = &order[..SPEC.top_k];
    let total = chosen.iter().map(|expert| scores[*expert]).sum::<f64>();
    let weights = chosen
        .iter()
        .map(|expert| scores[*expert] / total * f64::from(SPEC.routed_scale))
        .collect();
    (
        chosen.iter().map(|expert| u32::try_from(*expert).unwrap_or(u32::MAX)).collect(),
        weights,
        logits,
    )
}

#[test]
fn routes_by_biased_sigmoid_scores_with_renormalized_weights() -> Result<()> {
    for tokens in [1, 5, 8, 19] {
        route(tokens)?;
    }
    Ok(())
}

fn route(tokens: usize) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let router = SigmoidRouter::compile(backend.compiler(), SPEC)?;
    let input = pattern(tokens * SPEC.hidden, 1, 1.0);
    let weight = pattern(SPEC.experts * SPEC.hidden, 2, 0.2);
    let bias = pattern(SPEC.experts, 3, 0.3).into_iter().map(bf16::to_f32).collect::<Vec<_>>();
    let mut logits = backend.pool().allocate(backend.stream(), tokens * SPEC.experts)?;
    let mut selected = backend.pool().allocate(backend.stream(), tokens * SPEC.top_k)?;
    let mut weights = backend.pool().allocate(backend.stream(), tokens * SPEC.top_k)?;
    router.route(
        backend.stream(),
        tokens,
        &copy(&backend, &input)?,
        &copy(&backend, &weight)?,
        &copy(&backend, &bias)?,
        &mut SigmoidRouterOutputs {
            logits: &mut logits,
            selected: &mut selected,
            weights: &mut weights,
        },
    )?;
    let (logits, selected, weights) =
        (read(&backend, &logits)?, read(&backend, &selected)?, read(&backend, &weights)?);
    for token in 0..tokens {
        let (expected_selected, expected_weights, expected_logits) =
            reference(&input, &weight, &bias, token);
        for (actual, expected) in logits[token * SPEC.experts..].iter().zip(&expected_logits) {
            assert!((f64::from(*actual) - expected).abs() < 1e-3, "logit {actual} vs {expected}");
        }
        let range = token * SPEC.top_k..(token + 1) * SPEC.top_k;
        assert_eq!(selected[range.clone()], expected_selected[..], "token {token}");
        for (actual, expected) in weights[range].iter().zip(&expected_weights) {
            assert!((f64::from(actual.to_f32()) - expected).abs() < expected * 0.01);
        }
    }
    Ok(())
}

#[test]
fn squares_the_positive_part() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let activation = ReluSquaredBf16::compile(backend.compiler())?;
    let values = [-2.0_f32, -0.5, 0.0, 0.5, 3.0].map(bf16::from_f32);
    let mut output = backend.pool().allocate(backend.stream(), values.len())?;
    activation.execute(backend.stream(), &copy(&backend, &values)?, &mut output, values.len())?;
    let expected = [0.0_f32, 0.0, 0.0, 0.25, 9.0].map(bf16::from_f32);
    assert_eq!(read(&backend, &output)?, expected);
    Ok(())
}
