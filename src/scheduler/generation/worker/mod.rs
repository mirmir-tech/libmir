use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::mpsc::{Receiver, TryRecvError},
};

use runtime::{backend::ModelHandle, kv::BlockId, scheduler::SchedulerConfig};

use self::admission::{completion_wave_rows, prefill_wave_limit};
use super::{
    ActivePrefill, Command, PendingDecode, PendingPrefill,
    completion::{complete_decode, complete_decode_errors, complete_prefill_errors},
};
use crate::{Engine, engine::PrefillExecutionProfile};

mod admission;
mod budget;
mod finish;
mod handoff;
mod telemetry;

pub(super) struct Worker {
    engine: Engine,
    model: ModelHandle,
    config: SchedulerConfig,
    commands: Receiver<Command>,
    decode: VecDeque<PendingDecode>,
    prefill: VecDeque<PendingPrefill>,
    active_decode: HashMap<uuid::Uuid, Vec<BlockId>>,
    active_prefill: Option<ActivePrefill>,
    prefill_profile: PrefillExecutionProfile,
    prefill_handoff: handoff::PrefillHandoff,
    stopping: bool,
}

impl Worker {
    pub(super) fn new(
        engine: Engine,
        model: ModelHandle,
        config: SchedulerConfig,
        commands: Receiver<Command>,
        prefill_profile: PrefillExecutionProfile,
    ) -> Self {
        Self {
            engine,
            model,
            config,
            commands,
            decode: VecDeque::new(),
            prefill: VecDeque::new(),
            active_decode: HashMap::new(),
            active_prefill: None,
            prefill_profile,
            prefill_handoff: handoff::PrefillHandoff::default(),
            stopping: false,
        }
    }

    pub(super) fn run(mut self) {
        loop {
            if self.stopping {
                self.fail_all("accelerator generation worker stopped");
                return;
            }
            if !self.has_work() {
                match self.commands.recv() {
                    Ok(command) => self.admit(command),
                    Err(_) => self.stopping = true,
                }
            }
            self.drain_commands();
            if self.stopping {
                self.fail_all("accelerator generation worker stopped");
                return;
            }
            self.collect_decode_admission();
            self.collect_prefill_admission();
            self.prepare_prefill();
            if self.has_executable_work() {
                self.execute_step();
            } else if self.has_work() {
                match self.commands.recv() {
                    Ok(command) => self.admit(command),
                    Err(_) => self.stopping = true,
                }
            }
        }
    }

    fn has_work(&self) -> bool {
        !self.decode.is_empty() || !self.prefill.is_empty() || self.active_prefill.is_some()
    }

    fn has_executable_work(&self) -> bool {
        !self.decode.is_empty() || self.active_prefill.is_some()
    }

    fn admit(&mut self, command: Command) {
        match command {
            Command::Decode(mut pending) => {
                self.resolve_prefill_handoff(pending.sequence.session_id);
                let blocks = pending.sequence.block_table.blocks().to_vec();
                pending.newly_active =
                    self.active_decode.insert(pending.sequence.session_id, blocks).is_none();
                self.decode.push_back(pending);
            },
            Command::Prefill(pending) => self.prefill.push_back(pending),
            Command::Release(session) => {
                self.resolve_prefill_handoff(session);
                self.active_decode.remove(&session);
            },
            Command::Stop => self.stopping = true,
        }
    }

    fn drain_commands(&mut self) {
        loop {
            match self.commands.try_recv() {
                Ok(command) => self.admit(command),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.stopping = true;
                    return;
                },
            }
        }
    }

    fn prepare_prefill(&mut self) {
        if self.prefill_handoff_active() || self.active_prefill.is_some() || self.prefill.is_empty()
        {
            return;
        }
        self.prioritize_prefill();
        let available = self.prefill.len().min(self.prefill_admission_limit());
        let max_prompt_tokens = self
            .prefill
            .iter()
            .take(available)
            .map(|pending| pending.request.prompt_tokens.len())
            .max()
            .unwrap_or(1);
        let max_prefill_tokens = self.prefill_work_tokens(available);
        let resident_wave_rows = self.resident_prefill_rows(available);
        let wave_limit = prefill_wave_limit(
            self.config.max_batch_requests,
            self.config.max_batch_tokens,
            max_prefill_tokens,
            self.prefill_profile,
            resident_wave_rows,
        );
        if wave_limit == 0 {
            return;
        }
        let count = completion_wave_rows(available, wave_limit);
        let requests = self.prefill.drain(..count).collect::<Vec<_>>();
        let oldest_queue = requests.iter().map(|pending| pending.enqueued.elapsed()).max();
        let backend_requests =
            requests.iter().map(|pending| pending.request.clone()).collect::<Vec<_>>();
        let mut report = |row: usize, event| requests[row].response.report(event);
        match self.engine.prepare_generation_prefill(&backend_requests, &mut report) {
            Ok(batch) => {
                telemetry::trace_prefill_cohort(
                    self,
                    &requests,
                    wave_limit,
                    max_prompt_tokens,
                    max_prefill_tokens,
                    resident_wave_rows,
                    oldest_queue.unwrap_or_default(),
                );
                self.active_prefill = Some(ActivePrefill { batch, requests });
            },
            Err(error) => complete_prefill_errors(requests, &error.to_string()),
        }
    }

    fn execute_step(&mut self) {
        let decode = self.take_decode_batch();
        let count = decode.len();
        let sequences = decode.iter().map(|pending| pending.sequence.clone()).collect();
        if self.active_prefill.is_none() {
            match self.engine.decode_sequences(&self.model, sequences) {
                Ok(outputs) => {
                    self.observe_decode(decode.len());
                    complete_decode(decode, outputs);
                },
                Err(error) => complete_decode_errors(decode, &error.to_string()),
            }
            return;
        }
        let budget = self.prefill_step_budget(count);
        let result = {
            let responses = self.active_prefill.as_ref().map(|active| {
                active
                    .requests
                    .iter()
                    .map(|pending| pending.response.clone())
                    .collect::<Vec<_>>()
            });
            let mut report = |row: usize, event| {
                if let Some(response) = responses.as_ref().and_then(|items| items.get(row)) {
                    response.report(event);
                }
            };
            self.engine.execute_generation_step(
                &self.model,
                sequences,
                self.active_prefill.as_mut().map(|active| &mut active.batch),
                budget,
                &mut report,
            )
        };
        match result {
            Ok(output) => self.complete_step(decode, output),
            Err(error) => {
                complete_decode_errors(decode, &error.to_string());
                self.fail_active_prefill(&error.to_string());
            },
        }
    }

    fn decode_limit(&self) -> usize {
        self.config.max_batch_requests.min(self.config.max_batch_tokens).max(1)
    }

    pub(super) fn observe_decode(&mut self, rows: usize) {
        self.observe_prefill_handoff_decode(rows);
    }

    fn prefill_admission_limit(&self) -> usize {
        self.config.max_batch_requests.max(1)
    }

    fn active_resident_tokens(&self) -> usize {
        self.active_decode
            .values()
            .flatten()
            .copied()
            .collect::<HashSet<_>>()
            .len()
            .saturating_mul(self.prefill_profile.block_tokens)
    }
}
