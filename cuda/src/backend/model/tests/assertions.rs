pub(super) fn assert_logits_close(
    actual: &[mircuda::bf16],
    expected: &[mircuda::bf16],
    maximum_rmse: f64,
) {
    assert_eq!(actual.len(), expected.len());
    let (error, reference) = actual.iter().zip(expected).fold(
        (0.0_f64, 0.0_f64),
        |(error, reference), (actual, expected)| {
            let actual = f64::from(actual.to_f32());
            let expected = f64::from(expected.to_f32());
            let difference = actual - expected;
            (difference.mul_add(difference, error), expected.mul_add(expected, reference))
        },
    );
    let normalized_rmse = (error / reference.max(f64::EPSILON)).sqrt();
    assert_eq!(
        maximum(actual),
        maximum(expected),
        "optimized CUDA plan changed greedy token; normalized logits RMSE: {normalized_rmse:.6}"
    );
    assert!(
        normalized_rmse < maximum_rmse,
        "grouped W4A4 normalized logits RMSE: {normalized_rmse:.6}"
    );
}

fn maximum(values: &[mircuda::bf16]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.to_f32().total_cmp(&right.1.to_f32()))
        .map(|(index, _)| index)
}
