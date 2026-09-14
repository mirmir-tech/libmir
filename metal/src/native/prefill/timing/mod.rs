use std::time::{Duration, Instant};

use runtime::backend::PrefillTimings;

/// Per-row wall time from backend submission through output materialization.
/// Execution includes shared packed work for every participating row; it must
/// not be summed across rows as device time. Gaps include worker queueing,
/// interleaved decode, other rows and the wait until the batch is collected.
pub(in crate::native) struct PrefillTiming {
    started: Instant,
    execution: Duration,
}

impl PrefillTiming {
    pub(in crate::native) fn new(started: Instant) -> Self {
        Self { started, execution: Duration::ZERO }
    }

    pub(in crate::native) fn record(&mut self, execution: Duration) {
        self.execution += execution;
    }

    pub(in crate::native) fn finish(self) -> (Duration, PrefillTimings) {
        self.finish_at(Instant::now())
    }

    fn finish_at(self, finished: Instant) -> (Duration, PrefillTimings) {
        let elapsed = finished.duration_since(self.started);
        debug_assert!(self.execution <= elapsed, "prefill execution spans overlap");
        (
            elapsed,
            PrefillTimings {
                backend_wait: elapsed.saturating_sub(self.execution),
                backend_execution: self.execution,
                ..PrefillTimings::default()
            },
        )
    }
}

#[cfg(test)]
mod tests;
