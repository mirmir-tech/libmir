use std::{
    collections::HashSet,
    sync::mpsc::RecvTimeoutError,
    time::{Duration, Instant},
};

use runtime::kv::{BlockId, BlockTable};

use super::Worker;
use crate::{engine::PrefillExecutionProfile, scheduler::generation::Command};

mod priority;

const PREFILL_QUIET_WAIT_MULTIPLIER: u64 = 150;
const PREFILL_HARD_WAIT_MULTIPLIER: u32 = 4;

impl Worker {
    pub(super) fn collect_decode_admission(&mut self) {
        self.collect_prefill_handoff();
        if self.decode.is_empty() {
            return;
        }
        let wait = crate::scheduler::decode_admission_wait(self.config.decode_batch_wait_us);
        let deadline = Instant::now() + wait;
        while self.decode_needs_more() && !self.stopping {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match self.commands.recv_timeout(remaining) {
                Ok(command) => self.admit(command),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => self.stopping = true,
            }
        }
    }

    fn decode_needs_more(&self) -> bool {
        let limit = self.decode_limit();
        let admitting = self.decode.iter().any(|pending| pending.newly_active);
        let active = self.active_decode.len().min(limit).max(1);
        let target = if admitting && active < limit {
            active + 1
        } else {
            active
        };
        self.decode.len() < target
    }

    pub(super) fn collect_prefill_admission(&mut self) {
        if self.prefill_handoff_active() || self.active_prefill.is_some() || self.prefill.is_empty()
        {
            return;
        }
        let quiet = Duration::from_micros(
            self.config.decode_batch_wait_us.saturating_mul(PREFILL_QUIET_WAIT_MULTIPLIER),
        );
        let started = Instant::now();
        let hard_deadline = started + quiet.saturating_mul(PREFILL_HARD_WAIT_MULTIPLIER);
        let mut full_window = self.prefill_profile.collect_long_prefill_window
            && self
                .prefill
                .iter()
                .any(|pending| pending.request.prompt_tokens.len() > self.config.max_batch_tokens);
        let mut quiet_deadline =
            prefill_admission_deadline(started, quiet, hard_deadline, full_window);
        while self.prefill.len() < self.prefill_admission_limit() && !self.stopping {
            let deadline = quiet_deadline.min(hard_deadline);
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match self.commands.recv_timeout(remaining) {
                Ok(command) => {
                    let extends_quiet = matches!(command, Command::Prefill(_));
                    self.admit(command);
                    if extends_quiet {
                        full_window |= self.prefill_profile.collect_long_prefill_window
                            && self.prefill.back().is_some_and(|pending| {
                                pending.request.prompt_tokens.len() > self.config.max_batch_tokens
                            });
                        quiet_deadline = prefill_admission_deadline(
                            Instant::now(),
                            quiet,
                            hard_deadline,
                            full_window,
                        );
                    }
                },
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => self.stopping = true,
            }
        }
    }

    pub(super) fn take_decode_batch(&mut self) -> Vec<crate::scheduler::generation::PendingDecode> {
        let limit = self.decode_limit();
        let mut selected =
            if self.active_prefill.is_none() || !self.prefill_profile.defer_new_decode {
                self.decode.drain(..self.decode.len().min(limit)).collect()
            } else {
                take_ready(&mut self.decode, limit, |pending| !pending.newly_active)
            };
        for pending in &mut selected {
            pending.scheduler_queue = pending.enqueued.elapsed();
        }
        selected
    }

    pub(super) fn resident_prefill_rows(&self, available: usize) -> usize {
        let resident = self.active_decode.values().flatten().copied().collect();
        let capacity_blocks =
            self.prefill_profile.resident_token_slots / self.prefill_profile.block_tokens.max(1);
        resident_wave_rows(
            resident,
            self.prefill.iter().take(available).map(|pending| &pending.request.block_table),
            capacity_blocks,
        )
    }
}

fn take_ready<T>(
    queue: &mut std::collections::VecDeque<T>,
    limit: usize,
    mut ready: impl FnMut(&T) -> bool,
) -> Vec<T> {
    let mut selected = Vec::with_capacity(limit);
    let mut deferred = std::collections::VecDeque::with_capacity(queue.len());
    while let Some(item) = queue.pop_front() {
        if selected.len() < limit && ready(&item) {
            selected.push(item);
        } else {
            deferred.push_back(item);
        }
    }
    *queue = deferred;
    selected
}

fn next_prefill_deadline(now: Instant, quiet: Duration, hard_deadline: Instant) -> Instant {
    (now + quiet).min(hard_deadline)
}

fn prefill_admission_deadline(
    now: Instant,
    quiet: Duration,
    hard_deadline: Instant,
    full_window: bool,
) -> Instant {
    if full_window {
        hard_deadline
    } else {
        next_prefill_deadline(now, quiet, hard_deadline)
    }
}

pub(super) fn completion_wave_rows(available: usize, wave_limit: usize) -> usize {
    let limit = wave_limit.max(1);
    if available <= limit {
        return available;
    }
    let tail = available % limit;
    if tail == 1 && limit > 2 {
        limit - 1
    } else {
        limit
    }
}

pub(super) fn prefill_wave_limit(
    max_batch_requests: usize,
    max_batch_tokens: usize,
    max_prefill_tokens: usize,
    profile: PrefillExecutionProfile,
    resident_wave_rows: usize,
) -> usize {
    let budget = max_batch_tokens.max(1);
    let row_tokens = if profile.limit_deep_prefill_waves {
        max_prefill_tokens.clamp(1, profile.completion_round_tokens.max(1))
    } else {
        max_prefill_tokens.clamp(1, profile.chunk_tokens.max(1))
    };
    let full_wave = (budget / row_tokens).clamp(1, max_batch_requests.max(1));
    if !profile.limit_deep_prefill_waves {
        return full_wave.min(resident_wave_rows);
    }
    full_wave.min(resident_wave_rows).min(profile.max_prefill_wave_rows)
}

pub(super) fn pending_prefill_tokens(
    prompt_tokens: usize,
    cached_tokens: usize,
    replay_tokens: usize,
    block_tokens: usize,
) -> usize {
    let cached = cached_tokens.min(prompt_tokens);
    let missing = prompt_tokens.saturating_sub(cached);
    let replay = if missing == 0 && replay_tokens == 0 {
        block_tokens.max(1).min(cached)
    } else {
        replay_tokens.min(cached)
    };
    missing.saturating_add(replay).max(1)
}

fn resident_wave_rows<'a>(
    mut resident: HashSet<BlockId>,
    tables: impl IntoIterator<Item = &'a BlockTable>,
    capacity_blocks: usize,
) -> usize {
    let mut rows = 0;
    for table in tables {
        resident.extend(table.blocks().iter().copied());
        if resident.len() > capacity_blocks {
            break;
        }
        rows += 1;
    }
    rows
}

#[cfg(test)]
mod tests;
