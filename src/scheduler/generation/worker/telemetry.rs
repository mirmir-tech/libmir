use std::time::Duration;

use super::Worker;
use crate::scheduler::generation::PendingPrefill;

pub(super) fn trace_prefill_cohort(
    worker: &Worker,
    requests: &[PendingPrefill],
    wave_limit: usize,
    max_prompt_tokens: usize,
    max_prefill_tokens: usize,
    resident_wave_rows: usize,
    oldest_queue: Duration,
) {
    let cached_rows = requests.iter().filter(|row| row.request.cached_tokens > 0).count();
    let deep_cached_rows = requests
        .iter()
        .filter(|row| row.request.cached_tokens > worker.config.max_batch_tokens)
        .count();
    let cached_tokens = requests.iter().map(|row| row.request.cached_tokens).sum::<usize>();
    let missing_tokens = requests
        .iter()
        .map(|row| row.request.prompt_tokens.len().saturating_sub(row.request.cached_tokens))
        .sum::<usize>();
    let max_cached_tokens = requests.iter().map(|row| row.request.cached_tokens).max().unwrap_or(0);
    tracing::debug!(
        rows = requests.len(),
        cached_rows,
        deep_cached_rows,
        cached_tokens,
        missing_tokens,
        max_cached_tokens,
        wave_limit,
        max_prompt_tokens,
        max_prefill_tokens,
        prefill_chunk_tokens = worker.prefill_profile.chunk_tokens,
        completion_round_tokens = worker.prefill_profile.completion_round_tokens,
        cached_prefix_checkpoint_replay_tokens =
            worker.prefill_profile.cached_prefix_checkpoint_replay_tokens,
        cached_prefix_completion_slack_tokens =
            worker.prefill_profile.cached_prefix_completion_slack_tokens,
        resident_token_slots = worker.prefill_profile.resident_token_slots,
        resident_wave_rows,
        active_resident_tokens = worker.active_resident_tokens(),
        limit_deep_prefill_waves = worker.prefill_profile.limit_deep_prefill_waves,
        oldest_queue_ms = oldest_queue.as_secs_f64() * 1_000.0,
        "prepared accelerator generation-worker prefill cohort"
    );
}
