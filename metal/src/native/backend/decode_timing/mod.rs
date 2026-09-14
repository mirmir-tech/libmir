use std::time::Instant;

use runtime::backend::DecodeTimings;

/// Split the backend call at worker entry. Execution is host wall time and
/// includes graph submission/materialization; it is not GPU device timing.
pub(super) fn finish(started: Instant, executing: Instant, finished: Instant) -> DecodeTimings {
    DecodeTimings {
        backend_wait: executing.duration_since(started),
        backend_execution: finished.duration_since(executing),
        ..DecodeTimings::default()
    }
}

#[cfg(test)]
mod tests;
