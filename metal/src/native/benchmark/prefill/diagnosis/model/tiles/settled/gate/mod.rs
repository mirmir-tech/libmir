#[derive(Clone, Copy, Debug)]
pub(super) struct Timing {
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum WarmupStatus {
    Warming,
    Ready,
    Exhausted,
}

pub(super) const MAX_ROUNDS: usize = 6;
const WINDOW: usize = 3;

/// Every round contains one observation per plan, in canonical plan order.
/// Check both phases independently so one improving metric cannot hide drift.
pub(super) fn warmup_status(rounds: &[[Timing; 2]]) -> WarmupStatus {
    let settled = rounds.len() >= WINDOW
        && (0..2).all(|plan| {
            let recent = &rounds[rounds.len() - WINDOW..];
            within(recent.iter().map(|pair| pair[plan].prefill_ms))
                && within(recent.iter().map(|pair| pair[plan].decode_ms))
        });
    if settled {
        WarmupStatus::Ready
    } else if rounds.len() >= MAX_ROUNDS {
        WarmupStatus::Exhausted
    } else {
        WarmupStatus::Warming
    }
}

fn within(values: impl Iterator<Item = f64>) -> bool {
    let mut min = f64::INFINITY;
    let mut max = 0.0_f64;
    for value in values {
        if !value.is_finite() || value <= 0.0 {
            return false;
        }
        min = min.min(value);
        max = max.max(value);
    }
    max <= min * 1.05
}

#[cfg(test)]
mod tests;
