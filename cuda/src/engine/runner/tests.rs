use std::{sync::Arc, thread};

use super::{QueueState, RunnerQueue, WorkClass};
use crate::{Error, Result};

#[test]
fn decode_overtakes_prefill_until_burst_limit() {
    let mut state = QueueState::default();
    let prefill = state.enqueue(WorkClass::Prefill);
    let decode = state.enqueue(WorkClass::Decode);
    assert!(state.can_admit(WorkClass::Decode, decode, 2));
    assert!(!state.can_admit(WorkClass::Prefill, prefill, 2));
    state.admit(WorkClass::Decode);
    state.active = false;
    let second_decode = state.enqueue(WorkClass::Decode);
    assert!(state.can_admit(WorkClass::Decode, second_decode, 2));
}

#[test]
fn prefill_runs_after_decode_burst() {
    let mut state = QueueState::default();
    let prefill = state.enqueue(WorkClass::Prefill);
    state.decode_streak = 2;
    let decode = state.enqueue(WorkClass::Decode);
    assert!(state.can_admit(WorkClass::Prefill, prefill, 2));
    assert!(!state.can_admit(WorkClass::Decode, decode, 2));
}

#[test]
fn waiting_decode_runs_between_prefill_steps() -> Result<()> {
    let queue = Arc::new(RunnerQueue::new(Vec::new(), 2));
    let mut first = queue.acquire_prefill()?;
    first.push("vision-0");

    let decode_queue = queue.clone();
    let decode = thread::spawn(move || -> Result<()> {
        decode_queue.acquire_decode()?.push("decode");
        Ok(())
    });
    wait_for_decode(&queue)?;
    drop(first);

    let mut second = queue.acquire_prefill()?;
    second.push("vision-1");
    drop(second);
    let Ok(decoded) = decode.join() else {
        return Err(Error::State("decode test thread panicked".into()));
    };
    decoded?;

    let Ok(events) = queue.runner.lock() else {
        return Err(Error::State("CUDA model runner lock is poisoned".into()));
    };
    assert_eq!(*events, ["vision-0", "decode", "vision-1"]);
    drop(events);
    Ok(())
}

fn wait_for_decode<T>(queue: &RunnerQueue<T>) -> Result<()> {
    for _attempt in 0..10_000 {
        let Ok(state) = queue.state.lock() else {
            return Err(Error::State("CUDA runner queue lock is poisoned".into()));
        };
        let waiting = state.waiting_decode;
        drop(state);
        if waiting != 0 {
            return Ok(());
        }
        thread::yield_now();
    }
    Err(Error::State("decode did not enter the runner queue".into()))
}
