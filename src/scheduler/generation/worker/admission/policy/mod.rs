use std::time::Duration;

use super::{Worker, prefill_quiet_wait};
use crate::engine::PrefillAdmissionPolicy;

impl Worker {
    pub(super) fn prefill_collection_wait(&self) -> Duration {
        let configured = prefill_quiet_wait(
            self.config.prefill_batch_wait_us,
            self.prefill.len(),
            self.prefill_admission_limit(),
        );
        collection_wait(
            self.prefill_profile.admission,
            configured,
            self.prefill.iter().map(|pending| pending.request.prompt_tokens.len()),
            !self.active_decode.is_empty()
                || !self.decode.is_empty()
                || self.prefill_cohort.is_some()
                || !self.completed_prefill.is_empty(),
        )
    }
}

fn collection_wait(
    policy: PrefillAdmissionPolicy,
    configured: Duration,
    mut prompts: impl Iterator<Item = usize>,
    active_generation: bool,
) -> Duration {
    // Use full prompt length even on cache hits; never lengthen explicit waits.
    if matches!(policy, PrefillAdmissionPolicy::ShortPrompt)
        && !active_generation
        && prompts.all(|tokens| (1..=128).contains(&tokens))
    {
        configured.min(Duration::from_millis(3))
    } else {
        configured
    }
}

#[cfg(test)]
mod tests;
