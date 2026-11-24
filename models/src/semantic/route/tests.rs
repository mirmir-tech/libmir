//! The host routing on logits whose selections and weights are worked by
//! hand.

use super::RoutedExpert;
use crate::{
    error::Result,
    semantic::{ActivationSpec, ExpertGroupLimit, RoutedExpertsSpec, RouterNormalization},
};

fn spec(router_normalization: RouterNormalization, top_k: usize) -> RoutedExpertsSpec {
    RoutedExpertsSpec {
        expert_count: 4,
        top_k,
        intermediate_size: 8,
        activation: ActivationSpec::SwiGlu,
        router_normalization,
        group_limit: None,
        output_scale: 2.0,
    }
}

fn experts(route: &[RoutedExpert]) -> Vec<usize> {
    route.iter().map(|selected| selected.expert).collect()
}

fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0)
}

const LOGITS: [f32; 4] = [0.5, 2.0, -1.0, 2.0];

#[test]
fn softmax_routers_renormalize_or_keep_the_probability() -> Result<()> {
    // Ties go to the lower index; both chosen logits are equal.
    let renormalized = spec(RouterNormalization::SoftmaxTopK, 2).route(&LOGITS, None)?;
    assert_eq!(experts(&renormalized), [1, 3]);
    assert!(renormalized.iter().all(|selected| close(selected.weight, 1.0)));
    let probability = spec(RouterNormalization::SoftmaxProbabilityTopK, 2).route(&LOGITS, None)?;
    let total: f32 = LOGITS.iter().map(|logit| (logit - 2.0).exp()).sum();
    assert!(probability.iter().all(|selected| close(selected.weight, 2.0 / total)));
    Ok(())
}

#[test]
fn a_biased_sigmoid_selects_by_the_bias_and_weighs_without_it() -> Result<()> {
    let bias = [3.0, 0.0, 0.0, 0.0];
    let route = spec(RouterNormalization::SigmoidBiasedTopK, 2).route(&LOGITS, Some(&bias))?;
    assert_eq!(experts(&route), [0, 1]);
    let sigmoid = |logit: f32| (1.0 + (-logit).exp()).recip();
    let total = sigmoid(0.5) + sigmoid(2.0);
    assert!(close(route[0].weight, 2.0 * sigmoid(0.5) / total));
    assert!(spec(RouterNormalization::SigmoidBiasedTopK, 2).route(&LOGITS, None).is_err());
    Ok(())
}

#[test]
fn groups_rank_by_their_best_scores() -> Result<()> {
    let mut grouped = spec(RouterNormalization::SoftmaxTopK, 1);
    grouped.group_limit = Some(ExpertGroupLimit { groups: 2, selected: 1 });
    // Group 0 holds experts 0 and 1, whose best logit ties group 1's.
    assert_eq!(experts(&grouped.route(&[2.0, 0.0, 2.0, 1.0], None)?), [0]);
    grouped.router_normalization = RouterNormalization::SigmoidBiasedTopK;
    // By their two best scores, group 1 wins.
    let bias = [0.0; 4];
    assert_eq!(experts(&grouped.route(&[2.0, 0.0, 2.0, 1.0], Some(&bias))?), [2]);
    Ok(())
}

#[test]
fn the_sparsemixer_weighs_each_round_by_its_band() -> Result<()> {
    let route = spec(RouterNormalization::SparseMixer { jitter_eps: 0.01 }, 2)
        .route(&[1.0, 4.0, 3.99, -2.0], None)?;
    assert_eq!(experts(&route), [1, 2]);
    // Round one's band holds 4.0 and 3.99; round two's 3.99 and 4.0 again,
    // since the band reads the unmasked logits but sums the masked ones.
    assert!(close(route[0].weight, 2.0 / (1.0 + (-0.01_f32).exp())));
    assert!(close(route[1].weight, 2.0));
    Ok(())
}

#[test]
fn an_input_scaled_router_weighs_by_the_sigmoid() -> Result<()> {
    let route = spec(RouterNormalization::SigmoidInputScaledTopK, 1).route(&LOGITS, None)?;
    assert_eq!(experts(&route), [1]);
    assert!(close(route[0].weight, 2.0 * (1.0 + (-2.0_f32).exp()).recip()));
    Ok(())
}
