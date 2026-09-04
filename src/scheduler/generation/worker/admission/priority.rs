use std::time::{Duration, Instant};

use super::{PREFILL_HARD_WAIT_MULTIPLIER, PREFILL_QUIET_WAIT_MULTIPLIER, Worker};
use crate::scheduler::generation::PendingPrefill;

pub(super) fn take_ready<T>(
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

impl Worker {
    pub(in crate::scheduler::generation::worker) fn prioritize_prefill(&mut self) {
        let Some(replay_tokens) = self.prefill_profile.cached_prefix_replay_tokens else {
            return;
        };
        let now = Instant::now();
        let max_age = Duration::from_micros(
            self.config
                .decode_batch_wait_us
                .saturating_mul(PREFILL_QUIET_WAIT_MULTIPLIER)
                .saturating_mul(u64::from(PREFILL_HARD_WAIT_MULTIPLIER)),
        );
        let block_tokens = self.prefill_profile.block_tokens;
        self.prefill.make_contiguous().sort_by_key(|pending| {
            prefill_priority_key(now, max_age, replay_tokens, block_tokens, pending)
        });
    }
}

pub(super) fn prefill_priority_key(
    now: Instant,
    max_age: Duration,
    replay_tokens: usize,
    block_tokens: usize,
    pending: &PendingPrefill,
) -> (u8, usize, Instant) {
    if now.saturating_duration_since(pending.enqueued) >= max_age {
        return (0, 0, pending.enqueued);
    }
    (
        1,
        super::pending_prefill_tokens(
            pending.request.prompt_tokens.len(),
            pending.request.cached_tokens,
            replay_tokens,
            block_tokens,
        ),
        pending.enqueued,
    )
}
