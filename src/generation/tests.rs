use runtime::backend::SamplingLogits;

use super::{sampling::sampling, *};

#[test]
fn keeps_greedy_selection_on_the_backend() {
    let settings = GenerationSettings {
        max_tokens: 16,
        min_tokens: 0,
        ignore_eos: false,
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
        min_tokens: 0,
        ignore_eos: false,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        repetition_penalty: 1.0,
    };
    assert!(matches!(sampling(settings, 32_000), SamplingLogits::Sample { .. }));
}

#[test]
fn honors_minimum_tokens_and_ignore_eos() {
    let mut settings = GenerationSettings {
        max_tokens: 16,
        min_tokens: 4,
        ignore_eos: false,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };
    assert!(!should_stop(settings, 3, 2, &[2]));
    assert!(should_stop(settings, 4, 2, &[2]));
    settings.ignore_eos = true;
    assert!(!should_stop(settings, 16, 2, &[2]));
}
