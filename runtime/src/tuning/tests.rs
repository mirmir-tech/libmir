use std::time::Duration;

use super::{
    StartupBudget, TuningConfig, TuningMode, materially_faster, select_fastest_candidate,
    select_robust_candidate,
};

#[test]
fn defaults_to_bounded_startup_measurement() {
    let config = TuningConfig::default();
    assert_eq!(config.mode, TuningMode::Startup);
    assert_eq!(config.startup_budget_ms, 5_000);
    assert_eq!(config.minimum_improvement_bps, 300);
}

#[test]
fn startup_budget_saturates_after_measurement() {
    let mut budget = StartupBudget::new(Duration::from_millis(5));
    assert!(budget.available());
    budget.consume(Duration::from_millis(7));
    assert!(!budget.available());
    assert!(budget.remaining().is_zero());
}

#[test]
fn material_improvement_excludes_the_noise_margin() {
    assert!(!materially_faster(Duration::from_nanos(980), Duration::from_micros(1), 300));
    assert!(materially_faster(Duration::from_nanos(900), Duration::from_micros(1), 300));
}

#[test]
fn fastest_candidate_must_materially_beat_the_fallback() {
    let timings = [Duration::from_nanos(980), Duration::from_micros(1)];
    assert_eq!(select_fastest_candidate(0, 1, &timings, 300), 1);

    let timings = [Duration::from_nanos(900), Duration::from_micros(1)];
    assert_eq!(select_fastest_candidate(0, 1, &timings, 300), 0);
}

#[test]
fn invalid_measurement_index_retains_the_fallback() {
    assert_eq!(select_fastest_candidate(2, 0, &[Duration::from_micros(1)], 300), 0);
}

#[test]
fn robust_selection_rejects_a_shape_regression() {
    let timings = vec![
        vec![Duration::from_micros(80), Duration::from_micros(110)],
        vec![Duration::from_micros(100), Duration::from_micros(100)],
    ];
    assert_eq!(select_robust_candidate(1, &timings, 300), 1);

    let timings = vec![
        vec![Duration::from_micros(80), Duration::from_micros(100)],
        vec![Duration::from_micros(100), Duration::from_micros(105)],
    ];
    assert_eq!(select_robust_candidate(1, &timings, 300), 0);
}

#[test]
fn invalid_robust_measurements_retain_the_fallback() {
    let mismatched = vec![vec![Duration::from_micros(1)], vec![]];
    assert_eq!(select_robust_candidate(1, &mismatched, 300), 1);
    assert_eq!(select_robust_candidate(2, &mismatched, 300), 2);
}
