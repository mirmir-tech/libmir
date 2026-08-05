use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::mpsc::{Receiver, TryRecvError},
};

use runtime::{backend::ModelHandle, kv::BlockId, scheduler::SchedulerConfig};

use super::{
    ActivePrefill, Command, PendingDecode, PendingPrefill,
    completion::{complete_decode, complete_decode_errors},
};
use crate::{Engine, engine::PrefillExecutionProfile};

mod admission;
mod budget;
mod finish;
mod handoff;
mod prefill;
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
    prefill_cohort: Option<prefill::PrefillCohort>,
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
            prefill_cohort: None,
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
