mod response;

use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use runtime::{
    backend::{DecodeOutput, DecodeSequence, ModelHandle},
    scheduler::SchedulerConfig,
};

use self::response::DecodeResponse;
use crate::{Engine, Result};

pub struct DecodeCoordinator {
    engine: Engine,
    model: ModelHandle,
    config: SchedulerConfig,
    state: Mutex<State>,
    arrived: Condvar,
}

struct State {
    waiting: VecDeque<Pending>,
    running: bool,
    refill_rows: usize,
    refill_steps: usize,
}

struct Pending {
    sequence: DecodeSequence,
    response: Arc<DecodeResponse>,
    enqueued: Instant,
}

impl DecodeCoordinator {
    pub(super) fn new(engine: Engine, model: ModelHandle, config: SchedulerConfig) -> Self {
        Self {
            engine,
            model,
            config,
            state: Mutex::new(State {
                waiting: VecDeque::new(),
                running: false,
                refill_rows: 2,
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
        state.waiting.push_back(Pending {
            sequence,
            response,
            enqueued: Instant::now(),
        });
        let leader = !state.running;
        state.running = true;
        self.arrived.notify_one();
        Ok(leader)
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
        let target = state.target_rows(limit);
        let wait = state.refill_wait(self.config.decode_batch_wait_us);
        let waited = self
            .arrived
            .wait_timeout_while(state, wait, |state| state.waiting.len() < target);
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
        let sequences = pending.into_iter().map(|item| item.sequence).collect();
        match self.engine.decode_sequences(&self.model, sequences) {
            Ok(outputs) if outputs.len() == responses.len() => {
                for (response, output) in responses.into_iter().zip(outputs) {
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
            self.refill_rows = rows;
            self.refill_steps = 64;
        } else {
            self.refill_steps = self.refill_steps.saturating_sub(1);
        }
    }

    fn target_rows(&self, limit: usize) -> usize {
        if self.refill_steps == 0 {
            2.min(limit)
        } else {
            self.refill_rows.min(limit)
        }
    }

    fn refill_wait(&self, base_us: u64) -> Duration {
        Duration::from_micros(if self.refill_steps == 0 {
            base_us
        } else {
            base_us.saturating_mul(8)
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

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, time::Duration};

    use super::State;

    #[test]
    fn successful_batch_retains_a_bounded_refill_hint() {
        let mut state = State {
            waiting: VecDeque::new(),
            running: false,
            refill_rows: 2,
            refill_steps: 0,
        };
        assert_eq!(state.target_rows(16), 2);
        assert_eq!(state.refill_wait(200), Duration::from_micros(200));
        state.observe(4);
        assert_eq!(state.target_rows(16), 4);
        assert_eq!(state.refill_wait(200), Duration::from_micros(1_600));
        for _ in 0..64 {
            state.observe(1);
        }
        assert_eq!(state.target_rows(16), 2);
    }
}
