#[derive(Debug, serde::Serialize)]
pub(super) struct Difference {
    pub differing: usize,
    pub max_abs: f64,
    rms_error: f64,
    reference_rms: f64,
    relative_l2: Option<f64>,
    reference_max_abs: f64,
}

impl Difference {
    pub(super) fn between(actual: &[f32], reference: &[f32]) -> Self {
        assert_eq!(actual.len(), reference.len());
        assert!(!actual.is_empty());
        assert!(actual.iter().chain(reference).all(|v| v.is_finite()));
        let mut error_squared = 0.0_f64;
        let mut reference_squared = 0.0_f64;
        let mut max_abs = 0.0_f64;
        let mut reference_max_abs = 0.0_f64;
        let mut differing = 0;
        let mut count = 0.0_f64;
        for (&a, &b) in actual.iter().zip(reference) {
            differing += usize::from(a.to_bits() != b.to_bits());
            let (a, b) = (f64::from(a), f64::from(b));
            let difference = a - b;
            error_squared = difference.mul_add(difference, error_squared);
            reference_squared = b.mul_add(b, reference_squared);
            max_abs = max_abs.max(difference.abs());
            reference_max_abs = reference_max_abs.max(b.abs());
            count += 1.0;
        }
        Self {
            differing,
            max_abs,
            rms_error: (error_squared / count).sqrt(),
            reference_rms: (reference_squared / count).sqrt(),
            relative_l2: (reference_squared > 0.0)
                .then(|| (error_squared / reference_squared).sqrt()),
            reference_max_abs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Difference;

    #[test]
    fn distinguishes_absolute_error_from_relative_error_and_zero_reference() {
        let diff = Difference::between(&[3.0, 4.0], &[0.0, 4.0]);
        assert_eq!(diff.differing, 1);
        assert!((diff.max_abs - 3.0).abs() < f64::EPSILON);
        assert!(diff.relative_l2.is_some_and(|value| (value - 0.75).abs() < f64::EPSILON));
        let zero = Difference::between(&[1.0], &[0.0]);
        assert!(zero.relative_l2.is_none());
        assert!((zero.rms_error - 1.0).abs() < f64::EPSILON);
    }
}
