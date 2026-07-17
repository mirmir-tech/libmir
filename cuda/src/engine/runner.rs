use std::{
    ops::{Deref, DerefMut},
    sync::{Condvar, Mutex, MutexGuard},
    time::Instant,
};

use crate::{Error, Result};

pub(super) struct RunnerQueue<T> {
    state: Mutex<QueueState>,
    ready: Condvar,
    runner: Mutex<T>,
    decode_burst: usize,
}

pub(super) struct RunnerGuard<'a, T> {
    queue: &'a RunnerQueue<T>,
    runner: MutexGuard<'a, T>,
}

#[derive(Clone, Copy, Debug)]
enum WorkClass {
    Decode,
    Prefill,
}

#[derive(Default)]
struct QueueState {
    active: bool,
    next_decode: u64,
    serving_decode: u64,
    waiting_decode: usize,
    next_prefill: u64,
    serving_prefill: u64,
    waiting_prefill: usize,
    decode_streak: usize,
}

impl<T> RunnerQueue<T> {
    pub(super) fn new(runner: T, decode_burst: usize) -> Self {
        Self {
            state: Mutex::new(QueueState::default()),
            ready: Condvar::new(),
            runner: Mutex::new(runner),
            decode_burst: decode_burst.max(1),
        }
    }

    pub(super) fn acquire_decode(&self) -> Result<RunnerGuard<'_, T>> {
        self.acquire(WorkClass::Decode)
    }

    pub(super) fn acquire_prefill(&self) -> Result<RunnerGuard<'_, T>> {
        self.acquire(WorkClass::Prefill)
    }

    fn acquire(&self, class: WorkClass) -> Result<RunnerGuard<'_, T>> {
        let started = Instant::now();
        let Ok(mut state) = self.state.lock() else {
            return Err(Error::State("CUDA runner queue lock is poisoned".into()));
        };
        let ticket = state.enqueue(class);
        while !state.can_admit(class, ticket, self.decode_burst) {
            let waited = self.ready.wait(state);
            let Ok(current) = waited else {
                return Err(Error::State("CUDA runner queue wait is poisoned".into()));
            };
            state = current;
        }
        state.admit(class);
        let waiting_decode = state.waiting_decode;
        let waiting_prefill = state.waiting_prefill;
        drop(state);

        let Ok(runner) = self.runner.lock() else {
            self.release();
            return Err(Error::State("CUDA model runner lock is poisoned".into()));
        };
        tracing::debug!(
            class = ?class,
            wait_ms = started.elapsed().as_secs_f64() * 1_000.0,
            waiting_decode,
            waiting_prefill,
            "admitted CUDA runner work"
        );
        Ok(RunnerGuard { queue: self, runner })
    }

    fn release(&self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.active = false;
        self.ready.notify_all();
    }
}

impl QueueState {
    fn enqueue(&mut self, class: WorkClass) -> u64 {
        match class {
            WorkClass::Decode => {
                let ticket = self.next_decode;
                self.next_decode = self.next_decode.wrapping_add(1);
                self.waiting_decode += 1;
                ticket
            },
            WorkClass::Prefill => {
                let ticket = self.next_prefill;
                self.next_prefill = self.next_prefill.wrapping_add(1);
                self.waiting_prefill += 1;
                ticket
            },
        }
    }

    fn can_admit(&self, class: WorkClass, ticket: u64, burst: usize) -> bool {
        if self.active {
            return false;
        }
        match class {
            WorkClass::Decode => {
                ticket == self.serving_decode
                    && (self.waiting_prefill == 0 || self.decode_streak < burst)
            },
            WorkClass::Prefill => {
                ticket == self.serving_prefill
                    && (self.waiting_decode == 0 || self.decode_streak >= burst)
            },
        }
    }

    fn admit(&mut self, class: WorkClass) {
        self.active = true;
        match class {
            WorkClass::Decode => {
                self.serving_decode = self.serving_decode.wrapping_add(1);
                self.waiting_decode -= 1;
                self.decode_streak = self.decode_streak.saturating_add(1);
            },
            WorkClass::Prefill => {
                self.serving_prefill = self.serving_prefill.wrapping_add(1);
                self.waiting_prefill -= 1;
                self.decode_streak = 0;
            },
        }
    }
}

impl<T> Deref for RunnerGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.runner
    }
}

impl<T> DerefMut for RunnerGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.runner
    }
}

impl<T> Drop for RunnerGuard<'_, T> {
    fn drop(&mut self) {
        self.queue.release();
    }
}

#[cfg(test)]
mod tests {
    use super::{QueueState, WorkClass};

    #[test]
    fn decode_overtakes_prefill_until_burst_limit() {
        let mut state = QueueState::default();
        let prefill = state.enqueue(WorkClass::Prefill);
        let decode = state.enqueue(WorkClass::Decode);
        assert!(state.can_admit(WorkClass::Decode, decode, 2));
        assert!(!state.can_admit(WorkClass::Prefill, prefill, 2));
        state.admit(WorkClass::Decode);
        state.active = false;
        let second_decode = state.enqueue(WorkClass::Decode);
        assert!(state.can_admit(WorkClass::Decode, second_decode, 2));
    }

    #[test]
    fn prefill_runs_after_decode_burst() {
        let mut state = QueueState::default();
        let prefill = state.enqueue(WorkClass::Prefill);
        state.decode_streak = 2;
        let decode = state.enqueue(WorkClass::Decode);
        assert!(state.can_admit(WorkClass::Prefill, prefill, 2));
        assert!(!state.can_admit(WorkClass::Decode, decode, 2));
    }
}
