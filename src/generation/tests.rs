use super::*;

#[test]
fn keeps_greedy_selection_on_the_backend() {
    let settings = GenerationSettings {
        max_tokens: 16,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };
    assert_eq!(sampling(settings, 32_000), SamplingLogits::None);
}

#[test]
fn uses_device_top_p_when_top_k_is_bounded() {
    let settings = GenerationSettings {
        max_tokens: 16,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        repetition_penalty: 1.0,
    };
    assert!(matches!(sampling(settings, 32_000), SamplingLogits::Sample { .. }));
}
