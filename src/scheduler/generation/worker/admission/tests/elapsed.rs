use std::{
    io::Write,
    sync::{Arc, atomic::AtomicBool, mpsc},
};

use super::{Duration, Instant, pending, profile};
use crate::{
    Engine, RuntimeConfig,
    scheduler::generation::{Command, worker::Worker},
};

#[test]
fn expired_prefill_admission_does_not_start_another_collection_window()
-> Result<(), Box<dyn std::error::Error>> {
    let (mut worker, sender) = worker()?;
    let old = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .ok_or("monotonic clock too young")?;
    worker.prefill.push_back(pending(old, 0));
    // A command is available if the collector waits again. The expired window
    // must return to execution; normal worker maintenance drains it next turn.
    assert!(sender.send(Command::Stop).is_ok());
    worker.collect_prefill_admission();
    assert!(!worker.stopping, "already-aged prefill started another receive window");
    assert_eq!(worker.prefill.len(), 1);
    Ok(())
}

#[test]
fn settled_companions_retire_the_long_collection_window() -> Result<(), Box<dyn std::error::Error>>
{
    let (mut worker, sender) = worker()?;
    let _first = worker.preparing.announce();
    let _second = worker.preparing.announce();
    let arrival = Instant::now().checked_sub(Duration::from_millis(5)).ok_or("clock")?;
    worker.prefill.push_back(pending(arrival, 0));
    assert_eq!(worker.prefill_collection_wait(), Duration::from_millis(30));
    sender.send(Command::Prefill(pending(arrival, 0)))?;
    sender.send(Command::Stop)?;
    worker.collect_prefill_admission();
    assert_eq!(worker.prefill.len(), 2);
    assert_eq!(worker.prefill_collection_wait(), Duration::from_millis(3));
    assert!(
        !worker.stopping,
        "settled cohort continued receiving in the obsolete long window"
    );
    Ok(())
}

#[test]
#[ignore = "explicit host-only admission timing diagnostic"]
fn measures_aged_prefill_collection() -> Result<(), Box<dyn std::error::Error>> {
    let (mut worker, _sender) = worker()?;
    let mut elapsed = Vec::new();
    for _ in 0..16 {
        worker.prefill.clear();
        worker.prefill.push_back(pending(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .ok_or("monotonic clock too young")?,
            0,
        ));
        let started = Instant::now();
        worker.collect_prefill_admission();
        elapsed.push(started.elapsed().as_secs_f64() * 1000.0);
        assert!(!worker.stopping);
        assert_eq!(worker.prefill.len(), 1);
    }
    writeln!(std::io::stdout(), "aged_prefill_collection_ms: {elapsed:?}")?;
    Ok(())
}

#[test]
fn unsettled_or_active_work_retains_the_long_collection_window()
-> Result<(), Box<dyn std::error::Error>> {
    for active in [false, true] {
        let (mut worker, sender) = worker()?;
        let _first = worker.preparing.announce();
        let _second = worker.preparing.announce();
        let _third = (!active).then(|| worker.preparing.announce());
        if active {
            worker.active_decode.insert(uuid::Uuid::new_v4(), Vec::new());
        }
        let arrival = Instant::now().checked_sub(Duration::from_millis(5)).ok_or("clock")?;
        worker.prefill.push_back(pending(arrival, 0));
        sender.send(Command::Prefill(pending(arrival, 0)))?;
        sender.send(Command::Stop)?;
        worker.collect_prefill_admission();
        assert_eq!(worker.prefill.len(), 2);
        assert_eq!(worker.prefill_collection_wait(), Duration::from_millis(30));
        assert!(worker.stopping, "active or incomplete cohort lost its collection window");
    }
    Ok(())
}

pub(super) fn worker() -> crate::Result<(Worker, mpsc::Sender<Command>)> {
    let config = RuntimeConfig::default();
    let (sender, commands) = mpsc::channel();
    let worker = Worker::new(
        Engine::from_config(&config)?,
        pending(Instant::now(), 0).request.model,
        config.scheduler,
        commands,
        profile(16, 16, 1_000_000, false),
        Arc::new(AtomicBool::new(false)),
        crate::scheduler::PreparingRequests::default(),
    );
    Ok((worker, sender))
}

#[test]
#[ignore = "host-only settled admission latency; compare original and fixed window builds"]
fn measures_settled_prefill_collection() -> Result<(), Box<dyn std::error::Error>> {
    let (mut worker, sender) = worker()?;
    let mut elapsed = Vec::new();
    for _ in 0..16 {
        worker.prefill.clear();
        let _first = worker.preparing.announce();
        let _second = worker.preparing.announce();
        let arrival = Instant::now();
        worker.prefill.push_back(pending(arrival, 0));
        sender.send(Command::Prefill(pending(arrival, 0)))?;
        assert_eq!(worker.prefill_collection_wait(), Duration::from_millis(30));
        let started = Instant::now();
        worker.collect_prefill_admission();
        elapsed.push(started.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(worker.prefill.len(), 2);
        assert!(!worker.stopping);
    }
    writeln!(std::io::stdout(), "settled_prefill_collection_ms: {elapsed:?}")?;
    Ok(())
}
