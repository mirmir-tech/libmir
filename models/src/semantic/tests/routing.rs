use super::super::{ActivationSpec, ExpertGroupLimit, RoutedExpertsSpec, RouterNormalization};

fn routed(
    router_normalization: RouterNormalization,
    groups: usize,
    selected: usize,
) -> RoutedExpertsSpec {
    RoutedExpertsSpec {
        expert_count: 64,
        top_k: 6,
        intermediate_size: 1408,
        activation: ActivationSpec::SwiGlu { alpha: 1.0, clamp: None, up_shift: 0.0 },
        router_normalization,
        group_limit: Some(ExpertGroupLimit { groups, selected }),
        output_scale: 1.0,
    }
}

#[test]
fn expert_groups_split_the_experts_and_hold_the_selection() {
    use RouterNormalization::{SigmoidBiasedTopK, SoftmaxProbabilityTopK, UnitTopK};
    for (normalization, groups, selected) in [
        (SigmoidBiasedTopK, 8, 4),
        (SoftmaxProbabilityTopK, 8, 3),
        // One expert per group ranks by its probability alone.
        (SoftmaxProbabilityTopK, 64, 6),
    ] {
        assert!(routed(normalization, groups, selected).validate().is_ok());
    }
    for (normalization, groups, selected) in [
        // More groups selected than exist, or none.
        (SigmoidBiasedTopK, 8, 9),
        (SigmoidBiasedTopK, 8, 0),
        // Groups that do not split the experts evenly.
        (SigmoidBiasedTopK, 7, 4),
        // A sigmoid group needs two experts for its score.
        (SigmoidBiasedTopK, 64, 6),
        // One four-expert group cannot hold six experts.
        (SoftmaxProbabilityTopK, 16, 1),
        (UnitTopK, 8, 4),
        (SoftmaxProbabilityTopK, 1, 1),
    ] {
        assert!(
            routed(normalization, groups, selected).validate().is_err(),
            "{normalization:?} {groups}/{selected}"
        );
    }
}
