use std::{
    sync::{Condvar, Mutex},
    time::{Duration, Instant},
};

const CACHE_COHORT_WAIT_MULTIPLIER: u64 = 50;
const LONG_CACHE_COHORT_WAIT_MULTIPLIER: u64 = 1_000;

pub(super) struct CacheCohort {
    short_wait: Duration,
    long_wait: Duration,
    long_prefill_tokens: usize,
    state: Mutex<State>,
    ready: Condvar,
}

#[derive(Default)]
struct State {
    short: Window,
    long: Window,
}

#[derive(Default)]
struct Window {
    generation: u64,
    deadline: Option<Instant>,
}

impl State {
    const fn window(&self, long: bool) -> &Window {
        if long {
            &self.long
        } else {
            &self.short
        }
    }

    const fn window_mut(&mut self, long: bool) -> &mut Window {
        if long {
            &mut self.long
        } else {
            &mut self.short
        }
    }
}

impl CacheCohort {
    pub(super) fn new(decode_batch_wait_us: u64, max_batch_tokens: usize) -> Self {
        Self {
            short_wait: Duration::from_micros(
                decode_batch_wait_us.saturating_mul(CACHE_COHORT_WAIT_MULTIPLIER),
            ),
            long_wait: Duration::from_micros(
                decode_batch_wait_us.saturating_mul(LONG_CACHE_COHORT_WAIT_MULTIPLIER),
            ),
            long_prefill_tokens: max_batch_tokens,
            state: Mutex::new(State::default()),
            ready: Condvar::new(),
        }
    }

    pub(super) fn wait(&self, needs_eviction: bool, missing_tokens: usize) -> Duration {
        let long = missing_tokens > self.long_prefill_tokens;
        let wait = if long {
            self.long_wait
        } else {
            self.short_wait
        };
        if !needs_eviction || wait.is_zero() {
            return Duration::ZERO;
        }
        let started = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let generation = state.window(long).generation;
        let deadline =
            *state.window_mut(long).deadline.get_or_insert_with(|| Instant::now() + wait);
        loop {
            if state.window(long).generation != generation {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let window = state.window_mut(long);
                window.generation = window.generation.wrapping_add(1);
                window.deadline = None;
                self.ready.notify_all();
                break;
            }
            let waited = self
                .ready
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = waited.0;
        }
        drop(state);
        started.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_wait_does_not_join_an_existing_short_window() {
        let cohort = CacheCohort::new(20, 16);
        cohort
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .short
            .deadline = Some(Instant::now() + Duration::from_millis(1));

        let elapsed = cohort.wait(true, 17);

        assert!(elapsed >= Duration::from_millis(10), "long window lasted only {elapsed:?}");
    }
}
