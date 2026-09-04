use std::{
    collections::{HashSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

const CACHE_COHORT_WAIT_MULTIPLIER: u64 = 50;
const LONG_CACHE_COHORT_WAIT_MULTIPLIER: u64 = 1_000;

pub(super) struct CacheCohort {
    short_wait: Duration,
    long_wait: Duration,
    long_prefill_tokens: usize,
    block_tokens: usize,
    shared: Arc<Shared>,
}

struct Shared {
    state: Mutex<State>,
    ready: Condvar,
}

#[derive(Default)]
struct State {
    short: Window,
    long: Window,
    fills: HashSet<u64>,
}

pub enum FillClaim {
    Leader(Option<CacheFillGuard>),
    Retry(Duration),
}

pub struct CacheFillGuard {
    shared: Arc<Shared>,
    key: u64,
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
    pub(super) fn new(
        decode_batch_wait_us: u64,
        max_batch_tokens: usize,
        block_tokens: usize,
    ) -> Self {
        Self {
            short_wait: Duration::from_micros(
                decode_batch_wait_us.saturating_mul(CACHE_COHORT_WAIT_MULTIPLIER),
            ),
            long_wait: Duration::from_micros(
                decode_batch_wait_us.saturating_mul(LONG_CACHE_COHORT_WAIT_MULTIPLIER),
            ),
            long_prefill_tokens: max_batch_tokens,
            block_tokens: block_tokens.max(1),
            shared: Arc::new(Shared {
                state: Mutex::new(State::default()),
                ready: Condvar::new(),
            }),
        }
    }

    pub(super) fn claim_fill(
        &self,
        tokens: &[u32],
        checkpoints: &[usize],
        cached_tokens: usize,
    ) -> FillClaim {
        let checkpoint = checkpoints
            .iter()
            .copied()
            .filter(|checkpoint| *checkpoint <= tokens.len())
            .max()
            .unwrap_or(0);
        let complete_blocks = tokens
            .len()
            .saturating_sub(1)
            .checked_div(self.block_tokens)
            .unwrap_or(0)
            .saturating_mul(self.block_tokens);
        let prefix_tokens = if checkpoint > self.long_prefill_tokens {
            checkpoint
        } else {
            complete_blocks
        };
        if prefix_tokens <= self.long_prefill_tokens || cached_tokens >= prefix_tokens {
            return FillClaim::Leader(None);
        }
        let key = prefix_key(&tokens[..prefix_tokens]);
        let started = Instant::now();
        let mut state = self.shared.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.fills.insert(key) {
            return FillClaim::Leader(Some(CacheFillGuard {
                shared: Arc::clone(&self.shared),
                key,
            }));
        }
        while state.fills.contains(&key) {
            state =
                self.shared.ready.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        drop(state);
        FillClaim::Retry(started.elapsed())
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
        let mut state = self.shared.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
                self.shared.ready.notify_all();
                break;
            }
            let waited = self
                .shared
                .ready
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = waited.0;
        }
        drop(state);
        started.elapsed()
    }
}

impl Drop for CacheFillGuard {
    fn drop(&mut self) {
        {
            let mut state =
                self.shared.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            state.fills.remove(&self.key);
        }
        self.shared.ready.notify_all();
    }
}

fn prefix_key(tokens: &[u32]) -> u64 {
    let mut hash = DefaultHasher::new();
    tokens.hash(&mut hash);
    hash.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_wait_does_not_join_an_existing_short_window() {
        let cohort = CacheCohort::new(20, 16, 4);
        cohort
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .short
            .deadline = Some(Instant::now() + Duration::from_millis(1));

        let elapsed = cohort.wait(true, 17);

        assert!(elapsed >= Duration::from_millis(10), "long window lasted only {elapsed:?}");
    }

    #[test]
    fn identical_long_fill_waits_for_the_leader() {
        let cohort = Arc::new(CacheCohort::new(20, 16, 4));
        let claim = cohort.claim_fill(&[7; 19], &[17], 0);
        assert!(matches!(claim, FillClaim::Leader(Some(_))));
        let FillClaim::Leader(Some(leader)) = claim else {
            return;
        };
        let follower = Arc::clone(&cohort);
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            drop(sender.send(follower.claim_fill(&[7; 21], &[17], 0)));
        });
        assert!(receiver.recv_timeout(Duration::from_millis(10)).is_err());
        drop(leader);
        assert!(matches!(receiver.recv_timeout(Duration::from_secs(1)), Ok(FillClaim::Retry(_))));
        assert!(worker.join().is_ok());
    }
}
