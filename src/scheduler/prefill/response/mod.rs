use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
};

use runtime::{backend::PrefillOutput, progress::ProgressEvent};

use crate::Result;

pub(in crate::scheduler) struct PrefillResponse {
    state: Mutex<ResponseState>,
    ready: Condvar,
}

struct ResponseState {
    value: Option<Result<PrefillOutput>>,
    progress: VecDeque<ProgressEvent>,
}

impl PrefillResponse {
    pub(in crate::scheduler) fn new() -> Self {
        Self {
            state: Mutex::new(ResponseState { value: None, progress: VecDeque::new() }),
            ready: Condvar::new(),
        }
    }

    pub(in crate::scheduler) fn report(&self, event: ProgressEvent) {
        if let Ok(mut state) = self.state.lock() {
            state.progress.push_back(event);
            self.ready.notify_one();
        }
    }

    pub(in crate::scheduler) fn complete(&self, value: Result<PrefillOutput>) {
        if let Ok(mut state) = self.state.lock() {
            state.value = Some(value);
            self.ready.notify_one();
        }
    }

    pub(in crate::scheduler) fn wait(
        &self,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        self.wait_cancellable(progress, &crate::CancellationToken::default(), || Ok(()))
    }

    pub(in crate::scheduler) fn wait_cancellable(
        &self,
        progress: &mut dyn FnMut(ProgressEvent),
        cancellation: &crate::CancellationToken,
        mut wake: impl FnMut() -> Result<()>,
    ) -> Result<PrefillOutput> {
        let mut notified = false;
        loop {
            if cancellation.is_cancelled() && !notified {
                wake()?;
                notified = true;
            }
            let Ok(state) = self.state.lock() else {
                return Err(response_error());
            };
            let Ok((mut state, _)) =
                self.ready
                    .wait_timeout_while(state, crate::cancellation::POLL_INTERVAL, |state| {
                        state.value.is_none() && state.progress.is_empty()
                    })
            else {
                return Err(response_error());
            };
            let events: Vec<_> = state.progress.drain(..).collect();
            let value = state.value.take();
            drop(state);
            for event in events {
                progress(event);
            }
            if let Some(value) = value {
                return value;
            }
        }
    }
}

fn response_error() -> crate::Error {
    runtime::RuntimeError::Scheduler("prefill response lock is poisoned".into()).into()
}

#[cfg(test)]
mod tests;
