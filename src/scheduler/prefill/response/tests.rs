use std::{
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use runtime::progress::ProgressEvent;

use super::{PrefillResponse, response_error};

#[test]
fn progress_is_forwarded_before_completion() {
    let response = Arc::new(PrefillResponse::new());
    let waiting = response.clone();
    let (sent, received) = mpsc::channel();
    let waiter = thread::spawn(move || {
        let mut progress = |event| assert!(sent.send(event).is_ok());
        assert!(waiting.wait(&mut progress).is_err());
    });
    let event = ProgressEvent::prefill_tokens(256, 512);
    response.report(event.clone());
    assert_eq!(
        received.recv_timeout(Duration::from_secs(1)),
        Ok(event),
        "waiting request should receive progress before completion"
    );
    response.complete(Err(response_error()));
    assert!(waiter.join().is_ok());
}

#[test]
fn cancellation_wakes_idle_worker_but_waits_for_retirement_acknowledgement() {
    let response = Arc::new(PrefillResponse::new());
    let token = crate::CancellationToken::new();
    let waiter_token = token.clone();
    let waiting = response.clone();
    let (wake, awakened) = mpsc::channel();
    let (done, finished) = mpsc::channel();
    let waiter = thread::spawn(move || {
        let result = waiting.wait_cancellable(&mut |_| {}, &waiter_token, || {
            assert!(wake.send(()).is_ok());
            Ok(())
        });
        assert!(done.send(matches!(result, Err(crate::Error::Cancelled))).is_ok());
    });
    token.cancel();
    assert_eq!(awakened.recv_timeout(Duration::from_secs(1)), Ok(()));
    assert!(
        finished.recv_timeout(Duration::from_millis(30)).is_err(),
        "waiter released cache before worker acknowledgement"
    );
    assert!(awakened.try_recv().is_err(), "cancellation wake must be sent only once");
    response.complete(Err(crate::Error::Cancelled));
    assert_eq!(finished.recv_timeout(Duration::from_secs(1)), Ok(true));
    assert!(waiter.join().is_ok());
}
