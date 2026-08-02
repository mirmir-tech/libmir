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
        loop {
            let Ok(state) = self.state.lock() else {
                return Err(response_error());
            };
            let Ok(mut state) = self
                .ready
                .wait_while(state, |state| state.value.is_none() && state.progress.is_empty())
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
mod tests {
    use std::{
        sync::{Arc, mpsc},
        thread,
        time::Duration,
    };

    use runtime::progress::ProgressEvent;

    use super::{PrefillResponse, response_error};

    #[test]
    fn progress_is_forwarded_before_completion() {
        let response = Arc::new(PrefillResponse::new());
        let waiting = response.clone();
        let (sent, received) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let mut progress = |event| assert!(sent.send(event).is_ok());
            assert!(waiting.wait(&mut progress).is_err());
        });
        let event = ProgressEvent::prefill_tokens(256, 512);
        response.report(event.clone());
        assert_eq!(
            received.recv_timeout(Duration::from_secs(1)),
            Ok(event),
            "waiting request should receive progress before completion"
        );
        response.complete(Err(response_error()));
        assert!(waiter.join().is_ok());
    }
}
