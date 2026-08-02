use std::{path::PathBuf, time::Duration};

const BASIS_POINTS: u128 = 10_000;

/// Runtime behavior shared by accelerator execution tuners.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TuningMode {
    Disabled,
    Cached,
    #[default]
    Startup,
}

/// Common bounds and persistence policy for startup candidate selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TuningConfig {
    pub mode: TuningMode,
    pub cache_directory: Option<PathBuf>,
    pub startup_budget_ms: u64,
    pub warmup_iterations: u32,
    pub measurement_iterations: u32,
    pub minimum_improvement_bps: u16,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            mode: TuningMode::Startup,
            cache_directory: None,
            startup_budget_ms: 5_000,
            warmup_iterations: 1,
            measurement_iterations: 3,
            minimum_improvement_bps: 300,
        }
    }
}

/// Monotonic budget consumed by synchronized backend measurements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StartupBudget {
    remaining: Duration,
}

impl StartupBudget {
    #[must_use]
    pub const fn new(remaining: Duration) -> Self {
        Self { remaining }
    }

    #[must_use]
    pub const fn available(self) -> bool {
        !self.remaining.is_zero()
    }

    pub fn consume(&mut self, elapsed: Duration) {
        self.remaining = self.remaining.saturating_sub(elapsed);
    }

    #[must_use]
    pub const fn remaining(self) -> Duration {
        self.remaining
    }
}

/// Returns whether `candidate` beats `baseline` by the required margin.
#[must_use]
pub fn materially_faster(
    candidate: Duration,
    baseline: Duration,
    minimum_improvement_bps: u16,
) -> bool {
    let retained = BASIS_POINTS.saturating_sub(u128::from(minimum_improvement_bps));
    candidate.as_nanos().saturating_mul(BASIS_POINTS) < baseline.as_nanos().saturating_mul(retained)
}

/// Selects the fastest candidate only when it materially beats the fallback.
#[must_use]
pub fn select_fastest_candidate(
    fastest: usize,
    fallback: usize,
    timings: &[Duration],
    minimum_improvement_bps: u16,
) -> usize {
    let Some(candidate) = timings.get(fastest) else {
        return fallback;
    };
    let Some(baseline) = timings.get(fallback) else {
        return fallback;
    };
    if materially_faster(*candidate, *baseline, minimum_improvement_bps) {
        fastest
    } else {
        fallback
    }
}

/// Selects the best aggregate candidate without admitting a material
/// regression in any measured workload shape.
#[must_use]
pub fn select_robust_candidate(
    fallback: usize,
    timings: &[Vec<Duration>],
    minimum_improvement_bps: u16,
) -> usize {
    let Some(baseline) = timings.get(fallback) else {
        return fallback;
    };
    if baseline.is_empty() || timings.iter().any(|candidate| candidate.len() != baseline.len()) {
        return fallback;
    }
    let totals = timings
        .iter()
        .map(|candidate| candidate.iter().copied().fold(Duration::ZERO, Duration::saturating_add))
        .collect::<Vec<_>>();
    let Some((fastest, total)) = totals.iter().enumerate().min_by_key(|(_, total)| **total) else {
        return fallback;
    };
    if fastest == fallback
        || !materially_faster(*total, totals[fallback], minimum_improvement_bps)
        || timings[fastest].iter().zip(baseline).any(|(candidate, baseline)| {
            materially_faster(*baseline, *candidate, minimum_improvement_bps)
        })
    {
        fallback
    } else {
        fastest
    }
}

#[cfg(test)]
mod tests;
