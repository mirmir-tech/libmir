use std::{io::Write, sync::Arc};

use runtime::backend::{DecodeSequence, SamplingLogits};

use super::{Duration, Instant, elapsed::worker, pending, table};
use crate::scheduler::{
    generation::{Command, PendingDecode, PendingPrefill},
    response::DecodeResponse,
};

fn request(cancelled: bool) -> PendingPrefill {
    let mut request = pending(Instant::now(), 0);
    request.request.session_id = uuid::Uuid::new_v4();
    request.expects_decode = false;
    if cancelled {
        request.cancellation.cancel();
    }
    request
}

#[test]
fn admission_cancellation_retires_the_last_queued_prefill_without_another_receive()
-> crate::Result<()> {
    let (mut worker, sender) = worker()?;
    let cancelled = request(true);
    let response = cancelled.response.clone();
    worker.prefill.push_back(cancelled);
    assert!(sender.send(Command::Cancellation).is_ok());
    assert!(sender.send(Command::Stop).is_ok());
    worker.collect_prefill_admission();
    assert!(worker.prefill.is_empty(), "cancellation waited for the collection window");
    assert!(!worker.stopping, "empty collection waited for another command");
    assert!(matches!(response.wait(&mut |_| {}), Err(crate::Error::Cancelled)));
    Ok(())
}

#[test]
fn admission_cancellation_keeps_collecting_healthy_prefills() -> crate::Result<()> {
    let (mut worker, sender) = worker()?;
    worker.config.max_batch_requests = 3;
    let cancelled = request(true);
    let response = cancelled.response.clone();
    let first = request(false);
    let second = request(false);
    let third = request(false);
    let ids = [first.request.session_id, second.request.session_id, third.request.session_id];
    worker.prefill.push_back(cancelled);
    assert!(sender.send(Command::Cancellation).is_ok());
    // Keep a survivor queued when handling the notification, so collection
    // continues; reaching three healthy rows must not count the cancelled row.
    worker.prefill.push_back(first);
    assert!(sender.send(Command::Prefill(second)).is_ok());
    assert!(sender.send(Command::Prefill(third)).is_ok());
    assert!(sender.send(Command::Stop).is_ok());
    worker.collect_prefill_admission();
    assert_eq!(worker.prefill.iter().map(|p| p.request.session_id).collect::<Vec<_>>(), ids);
    assert!(!worker.stopping);
    assert!(matches!(response.wait(&mut |_| {}), Err(crate::Error::Cancelled)));
    Ok(())
}

#[test]
fn admission_cancellation_retires_prefill_while_collecting_decode() -> crate::Result<()> {
    let (mut worker, sender) = worker()?;
    let cancelled = request(true);
    let response = cancelled.response.clone();
    worker.prefill.push_back(cancelled);
    let id = uuid::Uuid::new_v4();
    worker.admit(Command::Decode(PendingDecode {
        sequence: DecodeSequence {
            session_id: id,
            token_id: 7,
            block_table: table(&[]),
            sampling_logits: SamplingLogits::None,
        },
        response: Arc::new(DecodeResponse::new()),
        enqueued: Instant::now(),
        scheduler_queue: Duration::ZERO,
        newly_active: true,
    }));
    assert!(sender.send(Command::Cancellation).is_ok());
    assert!(sender.send(Command::Stop).is_ok());
    worker.collect_decode_admission();
    assert!(worker.prefill.is_empty());
    assert_eq!(worker.decode.len(), 1);
    assert_eq!(worker.decode[0].sequence.session_id, id);
    assert!(matches!(response.wait(&mut |_| {}), Err(crate::Error::Cancelled)));
    Ok(())
}

#[test]
#[ignore = "explicit host-only admission cancellation timing"]
fn measures_admission_cancellation_wait() -> Result<(), Box<dyn std::error::Error>> {
    let (mut worker, sender) = worker()?;
    let mut times = Vec::new();
    for _ in 0..16 {
        worker.prefill.push_back(request(true));
        assert!(sender.send(Command::Cancellation).is_ok());
        let started = Instant::now();
        worker.collect_prefill_admission();
        worker.cancel_prefills();
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        assert!(worker.prefill.is_empty());
        assert!(!worker.stopping);
    }
    writeln!(std::io::stdout(), "admission_cancellation_ms: {times:?}")?;
    Ok(())
}
