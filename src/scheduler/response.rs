use std::sync::{Condvar, Mutex};

use runtime::backend::DecodeOutput;

use crate::{Result, scheduler::scheduler_error};

pub(super) struct DecodeResponse {
    value: Mutex<Option<std::result::Result<DecodeOutput, String>>>,
    ready: Condvar,
}

impl DecodeResponse {
    pub(super) const fn new() -> Self {
        Self {
            value: Mutex::new(None),
            ready: Condvar::new(),
        }
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
