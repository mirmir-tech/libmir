use super::{
    Worker,
    admission::{completion_wave_rows, prefill_wave_limit},
};
use crate::{
    engine::EnginePrefillCohort, scheduler::generation::completion::complete_prefill_errors,
};

pub(super) struct PrefillCohort {
    lease: EnginePrefillCohort,
    remaining: usize,
}

impl Worker {
    pub(super) fn prepare_prefill(&mut self) {
        if self.prefill_handoff_active()
            || self.active_prefill.is_some()
            || self.prefill.is_empty()
            || self.prefill_waits_for_decode()
        {
            return;
        }
        if let Err(error) = self
            .engine
            .refresh_prefill_memory_limits(&self.model, &mut self.prefill_profile)
        {
            tracing::warn!(%error, "could not refresh prefill memory limits");
        }
        if self.prefill_cohort.is_none() && !self.begin_prefill_cohort() {
            return;
        }
        let available = self.prefill.len().min(self.prefill_cohort_remaining());
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
        let mut requests = self.prefill.drain(..count).collect::<Vec<_>>();
        for pending in &mut requests {
            pending.scheduler_queue = pending.enqueued.elapsed();
        }
        let oldest_queue = requests.iter().map(|pending| pending.enqueued.elapsed()).max();
        let backend_requests =
            requests.iter().map(|pending| pending.request.clone()).collect::<Vec<_>>();
        let mut report = |row: usize, event| requests[row].response.report(event);
        let Some(cohort) = self.prefill_cohort.as_ref().map(|cohort| &cohort.lease) else {
            return;
        };
        match self
            .engine
            .prepare_generation_prefill(&backend_requests, Some(cohort), &mut report)
        {
            Ok(batch) => {
                self.advance_prefill_cohort(count);
                super::telemetry::trace_prefill_cohort(
                    self,
                    &requests,
                    wave_limit,
                    max_prompt_tokens,
                    max_prefill_tokens,
                    resident_wave_rows,
                    oldest_queue.unwrap_or_default(),
                );
                self.active_prefill =
                    Some(crate::scheduler::generation::ActivePrefill { batch, requests });
            },
            Err(error) => self.fail_prefill_cohort(requests, &error.to_string()),
        }
    }

    fn begin_prefill_cohort(&mut self) -> bool {
        self.prioritize_prefill();
        let count = prefill_cohort_rows(
            self.prefill.iter().map(|pending| self.prefill_completion_tokens(pending)),
            self.prefill_admission_limit(),
            self.prefill_profile.max_prefill_cohort_tokens,
        );
        let requests = self
            .prefill
            .iter()
            .take(count)
            .map(|pending| pending.request.clone())
            .collect::<Vec<_>>();
        match self.engine.prepare_generation_prefill_cohort(&requests) {
            Ok(lease) => {
                self.prefill_cohort = Some(PrefillCohort { lease, remaining: count });
                true
            },
            Err(error) => {
                let failed = self.prefill.drain(..count).collect();
                complete_prefill_errors(failed, &error.to_string());
                false
            },
        }
    }

    fn prefill_cohort_remaining(&self) -> usize {
        self.prefill_cohort.as_ref().map_or(0, |cohort| cohort.remaining)
    }

    fn advance_prefill_cohort(&mut self, count: usize) {
        let Some(cohort) = self.prefill_cohort.as_mut() else {
            return;
        };
        cohort.remaining = cohort.remaining.saturating_sub(count);
        if cohort.remaining == 0 {
            self.prefill_cohort = None;
        }
    }

    fn fail_prefill_cohort(
        &mut self,
        mut failed: Vec<crate::scheduler::generation::PendingPrefill>,
        message: &str,
    ) {
        let remaining = self.prefill_cohort_remaining().saturating_sub(failed.len());
        failed.extend(self.prefill.drain(..remaining));
        self.prefill_cohort = None;
        complete_prefill_errors(failed, message);
        self.fail_completed_prefill(message);
    }
}

fn prefill_cohort_rows(
    prompt_tokens: impl IntoIterator<Item = usize>,
    admission: usize,
    token_budget: usize,
) -> usize {
    let mut resident_tokens = 0_usize;
    let mut rows = 0_usize;
    for tokens in prompt_tokens.into_iter().take(admission.max(1)) {
        let tokens = tokens.max(1);
        if rows > 0 && resident_tokens.saturating_add(tokens) > token_budget.max(1) {
            break;
        }
        resident_tokens = resident_tokens.saturating_add(tokens);
        rows += 1;
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::prefill_cohort_rows;

    #[test]
    fn logical_cohort_groups_small_tails_and_bounds_cold_prompts() {
        let memory_tokens = 80 * 1_024;
        assert_eq!(prefill_cohort_rows([2_066; 10], 10, memory_tokens), 10);
        assert_eq!(prefill_cohort_rows([18_432; 10], 10, memory_tokens), 4);
        assert_eq!(prefill_cohort_rows([34_816; 10], 10, memory_tokens), 2);
        assert_eq!(prefill_cohort_rows([128 * 1_024; 10], 10, memory_tokens), 1);
    }
}
