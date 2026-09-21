use std::sync::{Condvar, Mutex};
#[cfg(any(feature = "cuda", feature = "metal"))]
use std::time::{Duration, Instant};

use runtime::backend::DecodeOutput;

use crate::{Result, scheduler::scheduler_error};

/// Waking a core from deep idle costs hundreds of microseconds, paid on every
/// decode step while the accelerator waits for the next token. The waiter
/// therefore wakes by timer shortly before the expected completion and polls.
#[cfg(any(feature = "cuda", feature = "metal"))]
const POLL_LEAD: Duration = Duration::from_micros(1_000);
#[cfg(any(feature = "cuda", feature = "metal"))]
const POLL_LIMIT: Duration = Duration::from_micros(3_000);

pub(super) struct DecodeResponse {
    value: Mutex<Option<std::result::Result<DecodeOutput, String>>>,
    ready: Condvar,
    #[cfg(any(feature = "cuda", feature = "metal"))]
    created: Instant,
}

impl DecodeResponse {
    pub(super) fn new() -> Self {
        Self {
            value: Mutex::new(None),
            ready: Condvar::new(),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            created: Instant::now(),
        }
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(super) fn elapsed(&self) -> Duration {
        self.created.elapsed()
    }

    /// Waits like [`Self::wait`], polling around `expected` after creation.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(super) fn wait_near(&self, expected: Duration) -> Result<DecodeOutput> {
        let Some(sleep) = expected.checked_sub(POLL_LEAD).filter(|sleep| !sleep.is_zero()) else {
            return self.wait();
        };
        {
            let Ok(mut value) = self.value.lock() else {
                return Err(scheduler_error("decode response lock is poisoned"));
            };
            while value.is_none() {
                let Some(remaining) = sleep.checked_sub(self.created.elapsed()) else {
                    break;
                };
                let Ok((next, _)) = self.ready.wait_timeout(value, remaining) else {
                    return Err(scheduler_error("decode response wait is poisoned"));
                };
                value = next;
            }
        }
        let limit = expected + POLL_LIMIT;
        while self.created.elapsed() < limit {
            if self.value.lock().is_ok_and(|value| value.is_some()) {
                break;
            }
            std::hint::spin_loop();
        }
        self.wait()
    }

    pub(super) fn complete(&self, value: std::result::Result<DecodeOutput, String>) {
        let Ok(mut current) = self.value.lock() else {
            return;
        };
        *current = Some(value);
        self.ready.notify_one();
    }

    pub(super) fn wait(&self) -> Result<DecodeOutput> {
        let Ok(mut value) = self.value.lock() else {
            return Err(scheduler_error("decode response lock is poisoned"));
        };
        while value.is_none() {
            let Ok(next) = self.ready.wait(value) else {
                return Err(scheduler_error("decode response wait is poisoned"));
            };
            value = next;
        }
        match value.take() {
            Some(Ok(output)) => Ok(output),
            Some(Err(message)) => Err(runtime::RuntimeError::Scheduler(message).into()),
            None => Err(scheduler_error("decode response is missing")),
        }
    }
}

#[cfg(all(test, any(feature = "cuda", feature = "metal")))]
mod tests {
    use std::{sync::Arc, thread, time::Duration};

    use runtime::backend::{DecodeOutput, TokenEvent};

    use super::DecodeResponse;

    fn complete_after(delay: Duration) -> Arc<DecodeResponse> {
        let response = Arc::new(DecodeResponse::new());
        let completer = Arc::clone(&response);
        thread::spawn(move || {
            thread::sleep(delay);
            completer.complete(Ok(DecodeOutput {
                event: TokenEvent {
                    token_id: Some(7),
                    text: String::new(),
                    finished: false,
                },
                logits: None,
                candidates: None,
                timings: None,
            }));
        });
        response
    }

    #[test]
    fn near_wait_returns_early_on_time_and_late_completions() {
        for (delay, expected) in [(1, 8), (8, 8), (30, 8), (5, 0)] {
            let response = complete_after(Duration::from_millis(delay));
            assert!(response.wait_near(Duration::from_millis(expected)).is_ok());
            assert!(response.elapsed() >= Duration::from_millis(delay));
        }
    }

    #[test]
    fn near_wait_reports_backend_errors() {
        let response = DecodeResponse::new();
        response.complete(Err("failed".into()));
        assert!(response.wait_near(Duration::from_millis(4)).is_err());
    }
}
