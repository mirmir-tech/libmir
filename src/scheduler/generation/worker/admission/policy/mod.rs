use std::time::Duration;

use super::{Worker, prefill_quiet_wait};
use crate::engine::PrefillAdmissionPolicy;

/// Quiet window once every announced request has reached the worker.
const SETTLED_COLLECTION_WAIT: Duration = Duration::from_millis(3);

impl Worker {
    pub(super) fn prefill_collection_wait(&self) -> Duration {
        let configured = prefill_quiet_wait(
            self.config.prefill_batch_wait_us,
            self.prefill.len(),
            self.prefill_admission_limit(),
        );
        let active_generation = !self.active_decode.is_empty()
            || !self.decode.is_empty()
            || self.prefill_cohort.is_some()
            || !self.completed_prefill.is_empty();
        // Companions can only come from requests still being prepared; without
        // any, a long quiet window merely delays an idle engine. Under active
        // generation the window is kept, because cohort timing there decides
        // the combined batch shapes and with them the qualified memory
        // peak.
        let configured = if active_generation || self.others_preparing() {
            configured
        } else {
            configured.min(SETTLED_COLLECTION_WAIT)
        };
        collection_wait(
            self.prefill_profile.admission,
            configured,
            self.prefill.iter().map(|pending| pending.request.prompt_tokens.len()),
            active_generation,
        )
    }
}

impl Worker {
    fn others_preparing(&self) -> bool {
        let received = self.prefill.len()
            + self.completed_prefill.len()
            + self.active_prefill.as_ref().map_or(0, |active| active.requests.len());
        self.preparing.count() > received
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
