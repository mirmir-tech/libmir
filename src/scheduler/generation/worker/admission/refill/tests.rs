use std::{
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use runtime::{
    backend::{ModelHandle, PrefillRequest, SamplingLogits},
    kv::{BlockId, BlockTable},
};

use crate::{
    CancellationToken, Engine, RuntimeConfig,
    engine::{EnginePrefillCohort, PrefillExecutionProfile},
    scheduler::{
        generation::{
            PendingPrefill,
            worker::{Worker, prefill::PrefillCohort},
        },
        prefill::PrefillResponse,
    },
};

fn worker() -> crate::Result<Worker> {
    let mut config = RuntimeConfig::default();
    config.scheduler.cached_prefill_policy = crate::CachedPrefillPolicy::InterleaveOneBlock;
    let (_, commands) = mpsc::channel();
    Ok(Worker::new(
        Engine::from_config(&config)?,
        ModelHandle {
            id: "test".into(),
            backend: "metal".into(),
        },
        config.scheduler,
        commands,
        PrefillExecutionProfile {
            chunk_tokens: 8,
            completion_round_tokens: 8,
            max_prefill_wave_rows: 2,
            max_prefill_wave_tokens: 64,
            max_prefill_cohort_tokens: 128,
            block_tokens: 16,
            resident_token_slots: 256,
            limit_deep_prefill_waves: true,
            cached_prefix_replay_tokens: Some(0),
            cached_prefix_checkpoint_replay_tokens: Some(0),
            cached_prefix_completion_slack_tokens: 0,
            defer_new_decode: true,
            interleave_prefill_decode: false,
        },
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
    ))
}

fn pending(cancelled: bool) -> PendingPrefill {
    let cancellation = CancellationToken::new();
    if cancelled {
        cancellation.cancel();
    }
    PendingPrefill {
        request: PrefillRequest {
            model: ModelHandle {
                id: "test".into(),
                backend: "metal".into(),
            },
            session_id: uuid::Uuid::new_v4(),
            prompt_tokens: vec![1; 32],
            cache_checkpoints: vec![],
            block_table: BlockTable::with_block_size(16),
            cached_tokens: 0,
            generation_tokens: None,
            sampling_logits: SamplingLogits::None,
        },
        response: Arc::new(PrefillResponse::new()),
        enqueued: Instant::now(),
        scheduler_queue: Duration::ZERO,
        expects_decode: false,
        cancellation,
    }
}

#[test]
fn cached_tail_can_enter_during_decode_but_not_a_cold_head() -> crate::Result<()> {
    let mut worker = worker()?;
    worker.config.max_batch_requests = 4;
    worker.config.max_batch_tokens = 64;
    worker.active_decode.insert(uuid::Uuid::new_v4(), vec![]);
    let mut tail = pending(false);
    tail.request.cached_tokens = 24;
    worker.prefill.push_back(tail);
    assert!(!worker.prefill_waits_for_decode());
    let response = worker.prefill[0].response.clone();
    worker.prefill[0].cancellation.cancel();
    assert!(worker.prefill_waits_for_decode());
    worker.cancel_prefills();
    assert!(matches!(response.wait(&mut |_| {}), Err(crate::Error::Cancelled)));
    assert_eq!(worker.active_decode.len(), 1);
    worker.prefill.push_back(pending(false));
    let mut tail = pending(false);
    tail.request.cached_tokens = 24;
    worker.prefill.push_back(tail);
    assert!(worker.prefill_waits_for_decode(), "must not jump a cold FIFO head");
    Ok(())
}

#[test]
fn cached_refill_respects_slots_replay_and_resident_capacity() -> crate::Result<()> {
    let mut worker = worker()?;
    worker.config.max_batch_requests = 2;
    worker.config.max_batch_tokens = 64;
    let id = uuid::Uuid::new_v4();
    worker.active_decode.insert(id, vec![BlockId(0), BlockId(1)]);
    let mut tail = pending(false);
    tail.request.cached_tokens = 24;
    worker.prefill.push_back(tail);
    assert!(worker.short_cached_refill_ready());
    worker.config.max_batch_requests = 1;
    assert!(!worker.short_cached_refill_ready());
    worker.config.max_batch_requests = 2;
    worker.config.max_batch_tokens = 8;
    assert!(!worker.short_cached_refill_ready());
    worker.config.max_batch_tokens = 64;
    worker.prefill_profile.resident_token_slots = 16;
    assert!(!worker.short_cached_refill_ready());
    worker.prefill_profile.resident_token_slots = 256;
    worker.prefill_profile.cached_prefix_checkpoint_replay_tokens = None;
    worker.prefill_profile.cached_prefix_replay_tokens = Some(32);
    assert!(!worker.short_cached_refill_ready());
    worker.prefill_profile.cached_prefix_replay_tokens = Some(0);
    worker.prefill_cohort = Some(PrefillCohort {
        lease: EnginePrefillCohort::default(),
        remaining: 1,
    });
    assert!(!worker.short_cached_refill_ready(), "must not split an existing lease");
    worker.prefill_cohort = None;
    worker.begin_prefill_handoff([id]);
    assert!(!worker.short_cached_refill_ready());
    Ok(())
}

#[test]
fn actual_refill_work_stays_bounded_if_a_snapshot_was_evicted() -> crate::Result<()> {
    let mut worker = worker()?;
    worker.config.max_batch_tokens = 2048;
    worker.active_decode.insert(uuid::Uuid::new_v4(), vec![]);
    assert_eq!(worker.prefill_step_budget(1), 16);
    assert_eq!(worker.prefill_step_budget(0), 16, "resident decode may be between commands");
    worker.active_decode.clear();
    assert_eq!(worker.prefill_step_budget(0), 2048);
    Ok(())
}

#[test]
fn default_keeps_backend_admission_and_step_budget() -> crate::Result<()> {
    let mut worker = worker()?;
    worker.config.cached_prefill_policy = crate::CachedPrefillPolicy::default();
    worker.config.max_batch_requests = 4;
    worker.config.max_batch_tokens = 2048;
    worker.active_decode.insert(uuid::Uuid::new_v4(), vec![]);
    let mut tail = pending(false);
    tail.request.cached_tokens = 24;
    worker.prefill.push_back(tail);
    assert!(worker.prefill_waits_for_decode());
    assert_eq!(worker.prefill_step_budget(1), 2047);
    Ok(())
}
