use super::stable;

#[test]
fn rejects_missing_invalid_and_unstable_timing_windows() {
    assert!(!stable(&[], 0.05));
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(!stable(&[100.0, invalid], 0.05));
    }
    assert!(stable(&[100.0, 105.0], 0.05));
    assert!(!stable(&[100.0, 105.01], 0.05));
}

#[test]
fn validation_keeps_the_original_baseline_anchor() {
    assert!(stable(&[105.0, 120.0], 0.15));
    assert!(!stable(&[100.0, 105.0, 120.0], 0.15));
}
