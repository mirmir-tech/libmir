use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use runtime::kv::{BlockId, BlockTable};

use super::{
    completion_wave_rows, next_prefill_deadline, pending_prefill_tokens,
    prefill_admission_deadline, prefill_wave_limit, resident_wave_rows, take_ready,
};
use crate::engine::PrefillExecutionProfile;

#[test]
fn prefill_quiet_window_extends_without_crossing_hard_deadline() {
    let started = Instant::now();
    let quiet = Duration::from_millis(10);
    let hard_deadline = started + Duration::from_millis(40);
    assert_eq!(
        next_prefill_deadline(started + Duration::from_millis(5), quiet, hard_deadline),
        started + Duration::from_millis(15)
    );
    assert_eq!(
        next_prefill_deadline(started + Duration::from_millis(35), quiet, hard_deadline),
        hard_deadline
    );
    assert_eq!(
        prefill_admission_deadline(started + Duration::from_millis(5), quiet, hard_deadline, true),
        hard_deadline
    );
}

#[test]
fn completion_wave_keeps_a_short_tail_in_one_cohort() {
    assert_eq!(completion_wave_rows(5, 8), 5);
    assert_eq!(completion_wave_rows(8, 8), 8);
    assert_eq!(completion_wave_rows(10, 8), 8);
}

#[test]
fn prefill_wave_targets_request_completion() {
    let capacity = 1_000_000;
    assert_eq!(
        prefill_wave_limit(16, 8_192, 10_240, profile(1_024, 8_192, capacity, true), 16),
        1
    );
    assert_eq!(
        prefill_wave_limit(16, 8_192, 102_048, profile(1_024, 8_192, capacity, true), 16),
        1
    );
    assert_eq!(prefill_wave_limit(16, 8_192, 2_048, profile(512, 512, capacity, true), 16), 16);
    assert_eq!(prefill_wave_limit(16, 8_192, 10_240, profile(512, 512, capacity, true), 16), 16);
    assert_eq!(
        prefill_wave_limit(16, 8_192, 10_240, profile(2_048, 2_048, capacity, true), 16),
        4
    );
    assert_eq!(
        prefill_wave_limit(10, 8_192, 6_144, profile(2_048, 2_048, capacity, true), 10),
        4
    );
    assert_eq!(
        prefill_wave_limit(2, 8_192, 102_048, profile(1_024, 8_192, capacity, true), 2),
        1
    );
    assert_eq!(
        prefill_wave_limit(16, 8_192, 4_096, profile(1_024, 8_192, capacity, true), 16),
        2
    );
    assert_eq!(prefill_wave_limit(16, 1, 1, profile(1, 1, 1, true), 1), 1);
}

#[test]
fn backend_caps_the_physical_prefill_wave() {
    let mut execution = profile(2_048, 2_048, 1_000_000, true);
    execution.max_prefill_wave_rows = 2;
    assert_eq!(prefill_wave_limit(10, 8_192, 2_066, execution, 10), 2);
    assert_eq!(prefill_wave_limit(10, 8_192, 4_100, execution, 10), 2);
}

#[test]
fn backend_can_retain_the_full_wave_for_restorable_prefixes() {
    let profile = profile(512, 8_192, 2_000_000, false);
    assert_eq!(prefill_wave_limit(16, 8_192, 100_000, profile, 16), 16);
}

#[test]
fn cached_prompts_schedule_only_the_missing_or_replayed_tail() {
    assert_eq!(pending_prefill_tokens(100_000, 0, 16, 16), 100_000);
    assert_eq!(pending_prefill_tokens(100_000, 99_984, 16, 16), 32);
    assert_eq!(pending_prefill_tokens(100_000, 100_000, 16, 16), 16);
    assert_eq!(pending_prefill_tokens(6_144, 4_080, 16, 16), 2_080);
    let deep = profile(1_024, 8_192, 333_536, true);
    assert_eq!(prefill_wave_limit(16, 8_192, 16, deep, 10), 10);
    assert_eq!(prefill_wave_limit(16, 8_192, 2_048, deep, 10), 4);
}

#[test]
fn resident_limit_counts_shared_blocks_once() {
    let active = HashSet::from([BlockId(9)]);
    let tables = [table(&[0, 1, 2]), table(&[0, 1, 3]), table(&[4, 5])];
    assert_eq!(resident_wave_rows(active, &tables, 6), 2);
}

#[test]
fn active_decode_overtakes_new_rows_without_reordering_the_tail() {
    let mut queue = [false, true, false, true].into_iter().collect();
    assert_eq!(take_ready(&mut queue, 1, |ready| *ready), [true]);
    assert_eq!(queue.into_iter().collect::<Vec<_>>(), [false, false, true]);
}

#[test]
fn recent_cached_prefill_overtakes_a_recent_cache_miss() {
    let now = Instant::now();
    let miss = pending(now.checked_sub(Duration::from_millis(5)).unwrap_or(now), 0);
    let hit = pending(now.checked_sub(Duration::from_millis(4)).unwrap_or(now), 99_984);
    let max_age = Duration::from_millis(40);
    assert!(
        super::priority::prefill_priority_key(now, max_age, 16, 16, &hit)
            < super::priority::prefill_priority_key(now, max_age, 16, 16, &miss)
    );
}

#[test]
fn aged_cache_miss_retains_fifo_priority() {
    let now = Instant::now();
    let miss = pending(now.checked_sub(Duration::from_millis(50)).unwrap_or(now), 0);
    let hit = pending(now.checked_sub(Duration::from_millis(4)).unwrap_or(now), 99_984);
    let max_age = Duration::from_millis(40);
    assert!(
        super::priority::prefill_priority_key(now, max_age, 16, 16, &miss)
            < super::priority::prefill_priority_key(now, max_age, 16, 16, &hit)
    );
}

fn profile(
    chunk_tokens: usize,
    completion_round_tokens: usize,
    resident_token_slots: usize,
    limit_deep_prefill_waves: bool,
) -> PrefillExecutionProfile {
    PrefillExecutionProfile {
        chunk_tokens,
        completion_round_tokens,
        max_prefill_wave_rows: usize::MAX,
        block_tokens: 16,
        resident_token_slots,
        limit_deep_prefill_waves,
        cached_prefix_replay_tokens: Some(1_525),
        cached_prefix_checkpoint_replay_tokens: Some(0),
        cached_prefix_completion_slack_tokens: 16,
        defer_new_decode: false,
        collect_long_prefill_window: false,
    }
}

fn table(blocks: &[u32]) -> BlockTable {
    let mut table = BlockTable::with_block_size(16);
    for block in blocks {
        table.push(BlockId(*block));
    }
    table
}

fn pending(
    enqueued: Instant,
    cached_tokens: usize,
) -> crate::scheduler::generation::PendingPrefill {
    use std::sync::Arc;

    use runtime::backend::{ModelHandle, PrefillRequest, SamplingLogits};

    crate::scheduler::generation::PendingPrefill {
        request: PrefillRequest {
            model: ModelHandle {
                id: "test".into(),
                backend: "test".into(),
            },
            session_id: uuid::Uuid::nil(),
            prompt_tokens: vec![0; 100_000],
            cache_checkpoints: Vec::new(),
            block_table: BlockTable::with_block_size(16),
            cached_tokens,
            sampling_logits: SamplingLogits::None,
        },
        response: Arc::new(crate::scheduler::prefill::PrefillResponse::new()),
        enqueued,
        expects_decode: true,
    }
}
