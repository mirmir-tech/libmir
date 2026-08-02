mod response;

use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use runtime::{
    backend::{ModelHandle, PrefillOutput, PrefillRequest},
    progress::ProgressEvent,
    scheduler::SchedulerConfig,
};

pub(in crate::scheduler) use self::response::PrefillResponse;
use super::step::GenerationStepState;
use crate::{Engine, Result};

const COHORT_WAIT_MULTIPLIER: u64 = 25;

/// Cohorts CUDA prefills so newly admitted sessions reach decode together.
pub struct PrefillCoordinator {
    engine: Engine,
    model: ModelHandle,
    config: SchedulerConfig,
    step: Arc<GenerationStepState>,
    state: Mutex<State>,
    arrived: Condvar,
}

struct State {
    waiting: VecDeque<Pending>,
    running: bool,
}

struct Pending {
    request: PrefillRequest,
    response: Arc<PrefillResponse>,
    enqueued: Instant,
}

impl PrefillCoordinator {
    pub(crate) fn new(
        engine: Engine,
        model: ModelHandle,
        config: SchedulerConfig,
        step: Arc<GenerationStepState>,
    ) -> Self {
        Self {
            engine,
            model,
            config,
            step,
            state: Mutex::new(State { waiting: VecDeque::new(), running: false }),
            arrived: Condvar::new(),
        }
    }

    pub(crate) fn submit(
        &self,
        request: PrefillRequest,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        if request.model.id != self.model.id || request.model.backend != self.model.backend {
            return Err(scheduler_error("prefill request targets another loaded model"));
        }
        let response = Arc::new(PrefillResponse::new());
        let leader = self.enqueue(request, response.clone())?;
        if leader {
            self.run(&response, progress);
        }
        response.wait(progress)
    }

    fn enqueue(&self, request: PrefillRequest, response: Arc<PrefillResponse>) -> Result<bool> {
        let Ok(mut state) = self.state.lock() else {
            return Err(scheduler_error("prefill admission lock is poisoned"));
        };
        state.waiting.push_back(Pending {
            request,
            response,
            enqueued: Instant::now(),
        });
        let leader = !state.running;
        state.running = true;
        self.arrived.notify_one();
        Ok(leader)
    }

    fn run(&self, leader_response: &Arc<PrefillResponse>, progress: &mut dyn FnMut(ProgressEvent)) {
        loop {
            let cohort = match self.collect() {
                Ok(cohort) => cohort,
                Err(error) => {
                    self.fail_waiting(&error.to_string());
                    return;
                },
            };
            self.execute(cohort, leader_response, progress);
            let Ok(mut state) = self.state.lock() else {
                self.fail_waiting("prefill admission lock is poisoned");
                return;
            };
            if state.waiting.is_empty() {
                state.running = false;
                return;
            }
        }
    }

    fn collect(&self) -> Result<Vec<Pending>> {
        let Ok(state) = self.state.lock() else {
            return Err(scheduler_error("prefill admission lock is poisoned"));
        };
        let wait = Duration::from_micros(
            self.config.decode_batch_wait_us.saturating_mul(COHORT_WAIT_MULTIPLIER),
        );
        let target = self.config.max_batch_requests.max(1);
        let Ok((mut state, _timeout)) = self
            .arrived
            .wait_timeout_while(state, wait, |state| state.waiting.len() < target)
        else {
            return Err(scheduler_error("prefill admission wait is poisoned"));
        };
        let count = state.waiting.len().min(target);
        Ok(state.waiting.drain(..count).collect())
    }

    fn execute(
        &self,
        cohort: Vec<Pending>,
        leader_response: &Arc<PrefillResponse>,
        progress: &mut dyn FnMut(ProgressEvent),
    ) {
        let rows = cohort.len();
        let step = self.step.plan();
        let oldest_queue = cohort
            .iter()
            .map(|pending| pending.enqueued.elapsed())
            .max()
            .unwrap_or_default();
        let requests = cohort.iter().map(|pending| pending.request.clone()).collect::<Vec<_>>();
        let mut report = |row: usize, event| {
            let pending = &cohort[row];
            if Arc::ptr_eq(&pending.response, leader_response) {
                progress(event);
            } else {
                pending.response.report(event);
            }
        };
        match self.engine.prefill_requests_with_progress(
            &requests,
            step.prefill_tokens,
            &mut report,
        ) {
            Ok(outputs) if outputs.len() == rows => {
                for (pending, output) in cohort.into_iter().zip(outputs) {
                    pending.response.complete(Ok(output));
                }
            },
            Ok(_) => complete_errors(cohort, "backend returned another prefill batch size"),
            Err(error) => complete_errors(cohort, &error.to_string()),
        }
        tracing::debug!(
            rows,
            decode_tokens = step.decode_tokens,
            prefill_tokens = step.prefill_tokens,
            oldest_queue_ms = oldest_queue.as_secs_f64() * 1_000.0,
            "admitted model prefill cohort"
        );
    }

    fn fail_waiting(&self, message: &str) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.running = false;
        let responses: Vec<Arc<PrefillResponse>> =
            state.waiting.drain(..).map(|pending| pending.response).collect();
        drop(state);
        for response in responses {
            response.complete(Err(scheduler_error(message)));
        }
    }
}

fn scheduler_error(message: &str) -> crate::Error {
    runtime::RuntimeError::Scheduler(message.into()).into()
}

fn complete_errors(cohort: Vec<Pending>, message: &str) {
    for pending in cohort {
        pending.response.complete(Err(scheduler_error(message)));
    }
}

#[cfg(test)]
mod tests {
    use super::COHORT_WAIT_MULTIPLIER;

    #[test]
    fn prefill_cohort_uses_a_stable_collection_window() {
        assert_eq!(COHORT_WAIT_MULTIPLIER, 25);
    }
}
