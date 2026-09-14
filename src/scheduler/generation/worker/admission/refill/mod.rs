use super::super::{Worker, budget::admission_replay_tokens};

impl Worker {
    pub(in crate::scheduler::generation::worker) fn short_cached_refill_ready(&self) -> bool {
        if self.config.cached_prefill_policy != crate::CachedPrefillPolicy::InterleaveOneBlock
            || self.prefill_profile.interleave_prefill_decode
            || self.active_decode.is_empty()
            || self.active_prefill.is_some()
            || self.prefill_cohort.is_some()
            || self.prefill_handoff_active()
            || self.active_decode.len() >= self.config.max_batch_requests
        {
            return false;
        }
        let Some(pending) = self.prefill.front() else {
            return false;
        };
        let cached = pending.request.cached_tokens;
        let prompt = pending.request.prompt_tokens.len();
        if pending.cancellation.is_cancelled() || cached == 0 || cached > prompt {
            return false;
        }
        let Some(fallback) = self.prefill_profile.cached_prefix_replay_tokens else {
            return false;
        };
        let (replay, _) = admission_replay_tokens(
            cached,
            fallback,
            self.prefill_profile.cached_prefix_checkpoint_replay_tokens,
        );
        // Do not discount completion slack: this is the whole per-step budget.
        let work = super::pending_prefill_tokens(
            prompt,
            cached,
            replay,
            self.prefill_profile.block_tokens,
        );
        work <= self.prefill_profile.block_tokens.max(1)
            && work <= self.config.max_batch_tokens.saturating_sub(self.active_decode.len())
            && self.resident_prefill_rows(1) == 1
    }
}

#[cfg(test)]
mod tests;
