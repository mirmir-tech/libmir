use std::collections::HashSet;

use super::Worker;
use crate::{
    engine::PrefillRefillPolicy, scheduler::generation::completion::complete_prefill_errors,
};

mod selection;
use selection::refill_rows;

impl Worker {
    pub(super) fn refill_prefill(&mut self) {
        if !matches!(self.prefill_profile.refill, PrefillRefillPolicy::ShortPrompt)
            || !self.prefill_profile.interleave_prefill_decode
            || self.prefill.is_empty()
            || self.prefill_cohort.is_some()
            || self.prefill_handoff_active()
        {
            return;
        }
        let Some(active) = &self.active_prefill else {
            return;
        };
        if !active.requests.iter().any(|row| row.request.prompt_tokens.len() > 128) {
            return;
        }
        let existing_short_tokens = active
            .requests
            .iter()
            .filter(|row| row.request.prompt_tokens.len() <= 128)
            .map(|row| row.request.prompt_tokens.len())
            .sum::<usize>();
        let short_budget = (self.config.max_batch_tokens.saturating_sub(self.active_decode.len())
            / 2)
        .saturating_sub(existing_short_tokens);
        let live = self
            .active_decode
            .keys()
            .copied()
            .chain(active.requests.iter().map(|row| row.request.session_id))
            .collect::<HashSet<_>>();
        let resident = self
            .active_decode
            .values()
            .flatten()
            .copied()
            .chain(
                active
                    .requests
                    .iter()
                    .flat_map(|row| row.request.block_table.blocks().iter().copied()),
            )
            .collect::<HashSet<_>>();
        let capacity =
            self.prefill_profile.resident_token_slots / self.prefill_profile.block_tokens.max(1);
        let count = refill_rows(
            self.prefill
                .iter()
                .map(|row| (row.request.prompt_tokens.len(), row.request.block_table.blocks())),
            self.config.max_batch_requests.saturating_sub(live.len()),
            short_budget,
            resident,
            capacity,
        );
        if count == 0 {
            return;
        }
        let requests = self
            .prefill
            .iter()
            .take(count)
            .map(|row| row.request.clone())
            .collect::<Vec<_>>();
        let responses = self
            .prefill
            .iter()
            .take(count)
            .map(|row| row.response.clone())
            .collect::<Vec<_>>();
        let mut progress = |row: usize, event| responses[row].report(event);
        let Some(active) = &mut self.active_prefill else {
            return;
        };
        match self
            .engine
            .extend_generation_prefill(&mut active.batch, &requests, &mut progress)
        {
            Ok(true) => {
                let mut incoming = self.prefill.drain(..count).collect::<Vec<_>>();
                for row in &mut incoming {
                    row.scheduler_queue = row.enqueued.elapsed();
                }
                active.requests.splice(..0, incoming);
                tracing::info!(
                    added_rows = count,
                    active_rows = active.requests.len(),
                    "admitted short requests at prefill boundary"
                );
            },
            Ok(false) => {},
            Err(error) => self.fail_refill(count, &live, &error.to_string()),
        }
    }

    fn fail_refill(&mut self, count: usize, live: &HashSet<uuid::Uuid>, message: &str) {
        // Preparation may restore prefixes; retire only new session ownership.
        let incoming = self.prefill.drain(..count).collect::<Vec<_>>();
        let sessions = incoming
            .iter()
            .map(|row| row.request.session_id)
            .filter(|session| !live.contains(session))
            .collect::<HashSet<_>>();
        let mut failure = message.to_owned();
        for session in sessions {
            if let Err(release) = self.engine.release_session(&self.model, session) {
                failure = release.to_string();
            }
        }
        complete_prefill_errors(incoming, &failure);
    }
}
