use mircuda::bf16;

pub(super) const RANK: usize = 10;

const MINIMUM_TOP1_PERCENT: f64 = 97.0;
const MINIMUM_TOPK_PERCENT: f64 = 95.0;
const MAXIMUM_NORMALIZED_RMSE: f64 = 0.03;
const MAXIMUM_ABSOLUTE_ERROR: f64 = 2.5;
const MAXIMUM_MEAN_KL: f64 = 0.003;

#[derive(Default)]
pub(super) struct Metrics {
    pub steps: usize,
    pub top1: usize,
    pub topk_overlap: usize,
    pub squared_error: f64,
    pub squared_reference: f64,
    pub maximum_error: f64,
    pub kl_divergence: f64,
}

impl Metrics {
    pub(super) fn observe(&mut self, expected: &[bf16], actual: &[bf16]) {
        assert_eq!(actual.len(), expected.len());
        let expected_top = top_k(expected);
        let actual_top = top_k(actual);
        self.steps += 1;
        self.top1 += usize::from(expected_top[0] == actual_top[0]);
        self.topk_overlap += expected_top.iter().filter(|token| actual_top.contains(token)).count();
        for (expected, actual) in expected.iter().zip(actual) {
            let expected = f64::from(expected.to_f32());
            let actual = f64::from(actual.to_f32());
            let difference = actual - expected;
            self.squared_error = difference.mul_add(difference, self.squared_error);
            self.squared_reference = expected.mul_add(expected, self.squared_reference);
            self.maximum_error = self.maximum_error.max(difference.abs());
        }
        self.kl_divergence += kl_divergence(expected, actual);
    }

    pub(super) fn validate(&self, mode: &str) {
        let top1_percent = ratio(self.top1, self.steps);
        let overlap_percent = ratio(self.topk_overlap, self.steps * RANK);
        let nrmse = (self.squared_error / self.squared_reference.max(f64::EPSILON)).sqrt();
        let mean_kl = self.kl_divergence / count(self.steps);
        assert!(
            top1_percent >= MINIMUM_TOP1_PERCENT,
            "{mode} top-1 agreement {top1_percent:.3}% is below {MINIMUM_TOP1_PERCENT:.3}%"
        );
        assert!(
            overlap_percent >= MINIMUM_TOPK_PERCENT,
            "{mode} top-{RANK} overlap {overlap_percent:.3}% is below {MINIMUM_TOPK_PERCENT:.3}%"
        );
        assert!(
            nrmse <= MAXIMUM_NORMALIZED_RMSE,
            "{mode} normalized RMSE {nrmse:.6} exceeds {MAXIMUM_NORMALIZED_RMSE:.6}"
        );
        assert!(
            self.maximum_error <= MAXIMUM_ABSOLUTE_ERROR,
            "{mode} maximum logit error {:.6} exceeds {MAXIMUM_ABSOLUTE_ERROR:.6}",
            self.maximum_error
        );
        assert!(
            mean_kl <= MAXIMUM_MEAN_KL,
            "{mode} mean KL {mean_kl:.6} exceeds {MAXIMUM_MEAN_KL:.6}"
        );
    }
}

fn top_k(values: &[bf16]) -> Vec<usize> {
    let mut top: Vec<usize> = Vec::with_capacity(RANK);
    for (index, value) in values.iter().enumerate() {
        let score = value.to_f32();
        let position = top
            .iter()
            .position(|&other| score > values[other].to_f32())
            .unwrap_or(top.len());
        if position < RANK {
            top.insert(position, index);
            top.truncate(RANK);
        }
    }
    top
}

fn kl_divergence(expected: &[bf16], actual: &[bf16]) -> f64 {
    let expected_max = maximum(expected);
    let actual_max = maximum(actual);
    let expected_sum = partition(expected, expected_max);
    let actual_sum = partition(actual, actual_max);
    expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| {
            let expected_logit = f64::from(expected.to_f32());
            let actual_logit = f64::from(actual.to_f32());
            let probability = (expected_logit - expected_max).exp() / expected_sum;
            probability
                * (expected_logit - expected_max - expected_sum.ln() - actual_logit
                    + actual_max
                    + actual_sum.ln())
        })
        .sum()
}

fn maximum(values: &[bf16]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(value.to_f32()))
        .fold(f64::NEG_INFINITY, f64::max)
}

fn partition(values: &[bf16], maximum: f64) -> f64 {
    values.iter().map(|value| (f64::from(value.to_f32()) - maximum).exp()).sum()
}

#[allow(clippy::cast_precision_loss)]
pub(super) fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 * 100.0 / denominator as f64
}

#[allow(clippy::cast_precision_loss)]
fn count(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_logits_have_exact_quality_metrics() {
        let logits = (0_u16..16).map(|value| bf16::from_f32(f32::from(value))).collect::<Vec<_>>();
        let mut metrics = Metrics::default();
        metrics.observe(&logits, &logits);

        assert_eq!(metrics.steps, 1);
        assert_eq!(metrics.top1, 1);
        assert_eq!(metrics.topk_overlap, RANK);
        assert!(metrics.squared_error.abs() < f64::EPSILON);
        assert!(metrics.maximum_error.abs() < f64::EPSILON);
        assert!(metrics.kl_divergence.abs() < 1.0e-12);
        metrics.validate("identical");
    }

    #[test]
    fn ranking_and_kl_ignore_a_constant_logit_shift() {
        let expected =
            (0_u16..16).map(|value| bf16::from_f32(f32::from(value))).collect::<Vec<_>>();
        let actual = (0_u16..16)
            .map(|value| bf16::from_f32(f32::from(value) + 4.0))
            .collect::<Vec<_>>();

        assert_eq!(top_k(&expected), top_k(&actual));
        assert!(kl_divergence(&expected, &actual).abs() < 1.0e-12);
    }
}
