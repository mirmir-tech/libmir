use models::generation::GenerationSettings;
use runtime::backend::SamplingLogits;

use super::{sampling::sampling, *};

#[test]
fn detects_only_sustained_generation_cycles() {
    let mut tokens = (0..168).collect::<Vec<_>>();
    let cycle = [7, 11, 13, 17, 19, 23, 29, 31];
    tokens.extend(cycle.repeat(3));
    assert_eq!(cycle::repeated_cycle(&tokens), Some(cycle.len()));
}

#[test]
fn ignores_short_cycles_and_a_single_full_repeat() {
    let short = [1, 2, 3, 4].repeat(10);
    assert_eq!(cycle::repeated_cycle(&short), None);
    let mut tokens = (0..48).collect::<Vec<_>>();
    tokens.extend([1, 2, 3, 4, 1, 2, 3, 4]);
    assert_eq!(cycle::repeated_cycle(&tokens), None);
}

#[test]
fn detects_a_repeated_phrase_without_adjacent_cycles() {
    let phrase = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157];
    let mut tokens = (0..96).collect::<Vec<_>>();
    tokens.extend(phrase);
    tokens.extend(200..216);
    tokens.extend(phrase);
    tokens.extend(300..316);
    tokens.extend(phrase);
    tokens.extend(400..416);
    tokens.extend(phrase);
    assert_eq!(cycle::repeated_cycle(&tokens), Some(phrase.len()));
}

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
fn uses_device_sampling_for_the_full_distribution() {
    let settings = GenerationSettings {
        max_tokens: 16,
        min_tokens: 0,
        ignore_eos: false,
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };
    assert!(matches!(
        sampling(settings, 201_088),
        SamplingLogits::Sample { top_k: 0, top_p: 1.0, .. }
    ));
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
