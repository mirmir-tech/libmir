const RANK: usize = 10;

pub(super) struct Metrics {
    steps: usize,
    top1: usize,
    topk_overlap: usize,
    squared_error: f64,
    squared_reference: f64,
    maximum_error: f64,
    kl_divergence: f64,
}

impl Metrics {
    pub(super) const fn new() -> Self {
        Self {
            steps: 0,
            top1: 0,
            topk_overlap: 0,
            squared_error: 0.0,
            squared_reference: 0.0,
            maximum_error: 0.0,
            kl_divergence: 0.0,
        }
    }

    pub(super) fn observe(&mut self, expected: &[f32], actual: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let expected_top = top_k(expected);
        let actual_top = top_k(actual);
        self.steps += 1;
        self.top1 += usize::from(expected_top[0] == actual_top[0]);
        self.topk_overlap += expected_top.iter().filter(|token| actual_top.contains(token)).count();
        for (expected, actual) in expected.iter().zip(actual) {
            let expected = f64::from(*expected);
            let actual = f64::from(*actual);
            let difference = actual - expected;
            self.squared_error = difference.mul_add(difference, self.squared_error);
            self.squared_reference = expected.mul_add(expected, self.squared_reference);
            self.maximum_error = self.maximum_error.max(difference.abs());
        }
        self.kl_divergence += kl_divergence(expected, actual);
    }

    pub(super) fn report(&self) -> String {
        format!(
            "steps={} top1={:.3}% top{RANK}_overlap={:.3}% nrmse={:.6} max_abs={:.6} mean_kl={:.6}",
            self.steps,
            ratio(self.top1, self.steps),
            ratio(self.topk_overlap, self.steps * RANK),
            (self.squared_error / self.squared_reference.max(f64::EPSILON)).sqrt(),
            self.maximum_error,
            self.kl_divergence / count(self.steps),
        )
    }
}

fn top_k(values: &[f32]) -> Vec<usize> {
    let mut top = Vec::with_capacity(RANK);
    for (index, score) in values.iter().copied().enumerate() {
        let position = top.iter().position(|&other| score > values[other]).unwrap_or(top.len());
        if position < RANK {
            top.insert(position, index);
            top.truncate(RANK);
        }
    }
    top
}

fn kl_divergence(expected: &[f32], actual: &[f32]) -> f64 {
    let expected_max = maximum(expected);
    let actual_max = maximum(actual);
    let expected_sum = partition(expected, expected_max);
    let actual_sum = partition(actual, actual_max);
    expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| {
            let expected = f64::from(*expected);
            let actual = f64::from(*actual);
            let probability = (expected - expected_max).exp() / expected_sum;
            probability
                * (expected - expected_max - expected_sum.ln() - actual
                    + actual_max
                    + actual_sum.ln())
        })
        .sum()
}

fn maximum(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(*value)).fold(f64::NEG_INFINITY, f64::max)
}

fn partition(values: &[f32], maximum: f64) -> f64 {
    values.iter().map(|value| (f64::from(*value) - maximum).exp()).sum()
}

#[allow(clippy::cast_precision_loss)]
fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 * 100.0 / denominator as f64
}

#[allow(clippy::cast_precision_loss)]
fn count(value: usize) -> f64 {
    value as f64
}
