#[cfg(any(feature = "cuda", feature = "metal"))]
mod generation;
mod model;
mod prefill;
mod response;
mod step;
#[cfg(test)]
mod tests;

use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

pub use model::{ModelCoordinator, PendingModelDecode};
pub use prefill::PrefillCoordinator;
use runtime::{
    backend::{DecodeOutput, DecodeSequence, ModelHandle},
    scheduler::SchedulerConfig,
};
pub use step::GenerationStepState;

use self::response::DecodeResponse;
use crate::{Engine, Result};

const REFILL_WAIT_MULTIPLIER: u64 = 25;

pub struct DecodeCoordinator {
    engine: Engine,
    model: ModelHandle,
    config: SchedulerConfig,
    step: Arc<GenerationStepState>,
    state: Mutex<State>,
    arrived: Condvar,
}

struct State {
    waiting: VecDeque<Pending>,
    active: HashSet<uuid::Uuid>,
    running: bool,
    refill_steps: usize,
}

struct Pending {
    sequence: DecodeSequence,
    response: Arc<DecodeResponse>,
    enqueued: Instant,
    newly_active: bool,
}

impl DecodeCoordinator {
    pub(super) fn new(
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
            state: Mutex::new(State {
                waiting: VecDeque::new(),
                active: HashSet::new(),
                running: false,
                refill_steps: 0,
            }),
            arrived: Condvar::new(),
        }
    }

    pub(super) fn submit(&self, sequence: DecodeSequence) -> Result<DecodeOutput> {
        let response = Arc::new(DecodeResponse::new());
        let leader = self.enqueue(sequence, response.clone())?;
        if leader {
            self.run();
        }
        response.wait()
    }

    fn enqueue(&self, sequence: DecodeSequence, response: Arc<DecodeResponse>) -> Result<bool> {
        let Ok(mut state) = self.state.lock() else {
            return Err(scheduler_error("decode admission lock is poisoned"));
        };
        let newly_active = state.active.insert(sequence.session_id);
        if newly_active {
            self.step.register_decode();
        }
        state.waiting.push_back(Pending {
            sequence,
            response,
            enqueued: Instant::now(),
            newly_active,
        });
        let leader = !state.running;
        state.running = true;
        self.arrived.notify_one();
        Ok(leader)
    }

    pub(super) fn release(&self, session_id: uuid::Uuid) {
        if let Ok(mut state) = self.state.lock() {
            if state.active.remove(&session_id) {
                self.step.release_decode();
            }
            self.arrived.notify_all();
        }
    }

    fn run(&self) {
        loop {
            let batch = match self.collect() {
                Ok(batch) => batch,
                Err(error) => {
                    self.fail_waiting(&error.to_string());
                    return;
                },
            };
            let rows = batch.len();
            self.execute(batch);
            let Ok(mut state) = self.state.lock() else {
                self.fail_waiting("decode admission lock is poisoned");
                return;
            };
            state.observe(rows);
            if state.waiting.is_empty() {
                state.running = false;
                return;
            }
        }
    }

    fn collect(&self) -> Result<Vec<Pending>> {
        let Ok(state) = self.state.lock() else {
            return Err(scheduler_error("decode admission lock is poisoned"));
        };
        let limit = self.limit();
        let wait = state.refill_wait(self.config.decode_batch_wait_us);
        let waited = self.arrived.wait_timeout_while(state, wait, |state| {
            let admitting = state.waiting.iter().any(|pending| pending.newly_active);
            state.waiting.len() < collection_target(state.active.len(), admitting, limit)
        });
        let Ok((mut state, _timeout)) = waited else {
            return Err(scheduler_error("decode admission wait is poisoned"));
        };
        let count = state.waiting.len().min(limit);
        Ok(state.waiting.drain(..count).collect())
    }

    fn execute(&self, pending: Vec<Pending>) {
        let oldest = pending.first().map_or(Duration::ZERO, |item| item.enqueued.elapsed());
        let rows = pending.len();
        let responses = pending.iter().map(|item| item.response.clone()).collect::<Vec<_>>();
        let queue = pending.iter().map(|item| item.enqueued.elapsed()).collect::<Vec<_>>();
        let sequences = pending.into_iter().map(|item| item.sequence).collect();
        match self.engine.decode_sequences(&self.model, sequences) {
            Ok(outputs) if outputs.len() == responses.len() => {
                for ((response, mut output), scheduler_queue) in
                    responses.into_iter().zip(outputs).zip(queue)
                {
                    if let Some(timings) = output.timings.as_mut() {
                        timings.scheduler_queue = scheduler_queue;
                    }
                    response.complete(Ok(output));
                }
            },
            Ok(_) => complete_errors(responses, "backend returned another decode batch size"),
            Err(error) => complete_errors(responses, &error.to_string()),
        }
        tracing::debug!(
            rows,
            capacity = self.limit(),
            occupancy_per_mille = rows.saturating_mul(1_000) / self.limit(),
            queue_delay_ms = oldest.as_secs_f64() * 1_000.0,
            "admitted model decode batch"
        );
    }

    fn limit(&self) -> usize {
        self.config.max_batch_requests.min(self.config.max_batch_tokens).max(1)
    }

    fn fail_waiting(&self, message: &str) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.running = false;
        let responses = state.waiting.drain(..).map(|item| item.response).collect();
        drop(state);
        complete_errors(responses, message);
    }
}

impl State {
    fn observe(&mut self, rows: usize) {
        if rows > 1 {
            self.refill_steps = 64;
        } else {
            self.refill_steps = self.refill_steps.saturating_sub(1);
        }
    }

    fn refill_wait(&self, base_us: u64) -> Duration {
        Duration::from_micros(if self.refill_steps == 0 {
            base_us
        } else {
            base_us.saturating_mul(REFILL_WAIT_MULTIPLIER)
        })
    }
}

fn complete_errors(responses: Vec<Arc<DecodeResponse>>, message: &str) {
    for response in responses {
        response.complete(Err(message.into()));
    }
}

fn scheduler_error(message: &str) -> crate::Error {
    runtime::RuntimeError::Scheduler(message.into()).into()
}

fn collection_target(active: usize, admitting: bool, limit: usize) -> usize {
    let active = active.min(limit).max(1);
    if admitting && active < limit {
        active + 1
    } else {
        active
    }
}
