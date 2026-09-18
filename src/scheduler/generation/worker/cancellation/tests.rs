use std::{
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use runtime::{
    backend::{ModelHandle, PrefillOutput, PrefillRequest, SamplingLogits},
    kv::BlockTable,
};

use super::super::{Worker, prefill::PrefillCohort};
use crate::{
    CancellationToken, Engine, RuntimeConfig,
    engine::{EnginePrefillCohort, PrefillExecutionProfile},
    scheduler::{generation::PendingPrefill, prefill::PrefillResponse},
};

fn worker() -> crate::Result<Worker> {
    let config = RuntimeConfig::default();
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
            refill: crate::engine::PrefillRefillPolicy::Closed,
            admission: crate::engine::PrefillAdmissionPolicy::Uniform,
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
fn queued_cancellation_preserves_cohort_count_and_order() -> crate::Result<()> {
    let mut worker = worker()?;
    let first = pending(true);
    let second = pending(false);
    let third = pending(true);
    let fourth = pending(false);
    let survivors = [second.request.session_id, fourth.request.session_id];
    let responses = [first.response.clone(), third.response.clone()];
    worker.prefill.extend([first, second, third, fourth]);
    worker.prefill_cohort = Some(PrefillCohort {
        lease: EnginePrefillCohort::default(),
        remaining: 2,
    });
    worker.cancel_prefills();
    assert_eq!(worker.prefill_cohort.as_ref().map(|cohort| cohort.remaining), Some(1));
    assert_eq!(
        worker.prefill.iter().map(|p| p.request.session_id).collect::<Vec<_>>(),
        survivors
    );
    for response in responses {
        assert!(matches!(response.wait(&mut |_| {}), Err(crate::Error::Cancelled)));
    }
    Ok(())
}

#[test]
fn cancelling_last_queued_wave_publishes_already_completed_sibling() -> crate::Result<()> {
    let mut worker = worker()?;
    let waiting = pending(true);
    let completed = pending(false);
    let response = completed.response.clone();
    worker.prefill.push_back(waiting);
    worker.prefill_cohort = Some(PrefillCohort {
        lease: EnginePrefillCohort::default(),
        remaining: 1,
    });
    worker.completed_prefill.push((
        completed,
        PrefillOutput {
            accepted_tokens: 32,
            next_token: Some(7),
            trace: None,
            logits: None,
            candidates: None,
            timings: None,
        },
    ));
    worker.cancel_prefills();
    assert!(worker.prefill.is_empty());
    assert!(worker.prefill_cohort.is_none());
    assert!(worker.completed_prefill.is_empty());
    assert_eq!(response.wait(&mut |_| {})?.next_token, Some(7));
    Ok(())
}

#[test]
fn cancellation_maintenance_preserves_an_existing_decode_handoff() -> crate::Result<()> {
    let mut worker = worker()?;
    worker.prefill.push_back(pending(false));
    worker.begin_prefill_handoff([uuid::Uuid::new_v4()]);
    assert!(worker.prefill_handoff_active());
    worker.cancel_prefills();
    assert!(worker.prefill_handoff_active());
    Ok(())
}
