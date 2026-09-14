use std::time::{Duration, Instant};

use super::{PREFILL_HARD_WAIT_MULTIPLIER, next_prefill_deadline};

/// Collection time is bounded by queue arrival, including time spent behind
/// active accelerator work. Entering the collector must not restart that
/// budget.
pub(super) struct PrefillWindow {
    quiet: Duration,
    hard_deadline: Instant,
    quiet_deadline: Instant,
}

impl PrefillWindow {
    pub(super) fn new(
        quiet: Duration,
        mut arrivals: impl Iterator<Item = Instant>,
    ) -> Option<Self> {
        let first = arrivals.next()?;
        let mut window = Self {
            quiet,
            hard_deadline: first + quiet.saturating_mul(PREFILL_HARD_WAIT_MULTIPLIER),
            quiet_deadline: first + quiet,
        };
        for arrival in arrivals {
            window.arrived(arrival);
        }
        Some(window)
    }

    pub(super) fn arrived(&mut self, arrival: Instant) {
        // Senders and priority sorting can expose timestamps out of order.
        self.hard_deadline = self
            .hard_deadline
            .min(arrival + self.quiet.saturating_mul(PREFILL_HARD_WAIT_MULTIPLIER));
        self.quiet_deadline = self
            .quiet_deadline
            .max(next_prefill_deadline(arrival, self.quiet, self.hard_deadline))
            .min(self.hard_deadline);
    }

    pub(super) fn remaining(&self, now: Instant) -> Duration {
        self.quiet_deadline.saturating_duration_since(now)
    }
}

#[cfg(test)]
mod tests;
