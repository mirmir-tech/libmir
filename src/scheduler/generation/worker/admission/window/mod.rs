use std::time::{Duration, Instant};

use super::{PREFILL_HARD_WAIT_MULTIPLIER, next_prefill_deadline};

/// Deadlines stay anchored to arrivals when a long prompt widens the window.
pub(super) struct PrefillWindow {
    quiet: Duration,
    oldest: Instant,
    latest: Instant,
}

impl PrefillWindow {
    pub(super) fn new(
        quiet: Duration,
        mut arrivals: impl Iterator<Item = Instant>,
    ) -> Option<Self> {
        let first = arrivals.next()?;
        let mut window = Self { quiet, oldest: first, latest: first };
        for arrival in arrivals {
            window.arrived(arrival);
        }
        Some(window)
    }

    pub(super) fn arrived(&mut self, arrival: Instant) {
        self.oldest = self.oldest.min(arrival);
        self.latest = self.latest.max(arrival);
    }

    pub(super) fn widen(&mut self, quiet: Duration) {
        self.quiet = self.quiet.max(quiet);
    }

    pub(super) fn remaining(&self, now: Instant) -> Duration {
        let hard = self.oldest + self.quiet.saturating_mul(PREFILL_HARD_WAIT_MULTIPLIER);
        next_prefill_deadline(self.latest, self.quiet, hard).saturating_duration_since(now)
    }
}

#[cfg(test)]
mod tests;
