use super::{prefill_schedule, step_model_id};

#[test]
fn empty_generation_step_is_rejected() {
    assert!(step_model_id(None, None).is_err());
}

#[test]
fn routed_prefill_caps_resident_tokens_before_streaming() {
    let routed = prefill_schedule(true, Some((80 * 1_024, 320 * 1_024)));
    assert_eq!(routed.max_wave_rows, usize::MAX);
    assert_eq!(routed.max_wave_tokens, 80 * 1_024);
    assert_eq!(routed.max_cohort_tokens, 320 * 1_024);
    assert!(!routed.interleave_decode);

    let dense = prefill_schedule(false, None);
    assert_eq!(dense.max_wave_rows, usize::MAX);
    assert_eq!(dense.max_wave_tokens, usize::MAX);
    assert_eq!(dense.max_cohort_tokens, usize::MAX);
    assert!(dense.interleave_decode);
}
