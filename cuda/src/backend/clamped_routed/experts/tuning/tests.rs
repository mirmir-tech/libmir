use super::*;

#[test]
fn numerical_gate_accepts_rounding_and_rejects_drift() {
    let reference = [1.0, -20.0, 0.0].map(bf16::from_f32);
    let close = [1.125, -20.125, 0.125].map(bf16::from_f32);
    let drift = [1.5, -20.0, 0.0].map(bf16::from_f32);
    assert!(compare(&reference, &close).equivalent);
    assert!(!compare(&reference, &drift).equivalent);
}

#[test]
fn numerical_gate_tolerates_only_rare_small_outliers() {
    let reference = vec![bf16::from_f32(4.75); 2 * OUTLIER_DENOMINATOR];
    let with = |outliers: usize, value: f32| {
        let mut candidate = reference.clone();
        candidate[..outliers].fill(bf16::from_f32(value));
        compare(&reference, &candidate).equivalent
    };
    // 4.906 is one ulp group past the 0.125 tolerance; two allowed here.
    assert!(with(2, 4.906));
    assert!(!with(3, 4.906));
    // Far past the tolerance is a bug however rare.
    assert!(!with(1, 6.0));
}
