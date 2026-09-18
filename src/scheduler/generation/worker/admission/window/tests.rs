use super::{Duration, Instant, PrefillWindow};

#[test]
fn work_already_waiting_for_the_accelerator_does_not_wait_again() -> Result<(), &'static str> {
    let arrival = Instant::now();
    let now = arrival + Duration::from_secs(1);
    let quiet = Duration::from_millis(30);
    let window = PrefillWindow::new(quiet, [arrival].into_iter()).ok_or("one request")?;
    assert_eq!(window.remaining(now), Duration::ZERO);
    Ok(())
}

#[test]
fn fresh_bursts_keep_the_existing_quiet_window() -> Result<(), &'static str> {
    let start = Instant::now();
    let quiet = Duration::from_millis(30);
    let mut window = PrefillWindow::new(quiet, [start].into_iter()).ok_or("one request")?;
    assert_eq!(window.remaining(start), quiet);
    let next = start + Duration::from_millis(20);
    window.arrived(next);
    assert_eq!(window.remaining(next), quiet);
    assert_eq!(window.remaining(next + quiet), Duration::ZERO);
    Ok(())
}

#[test]
fn a_recent_arrival_cannot_renew_the_oldest_requests_hard_deadline() -> Result<(), &'static str> {
    let start = Instant::now();
    let quiet = Duration::from_millis(30);
    let recent = start + Duration::from_millis(115);
    let window = PrefillWindow::new(quiet, [start, recent].into_iter()).ok_or("two requests")?;
    assert_eq!(window.remaining(recent), Duration::from_millis(5));
    Ok(())
}

#[test]
fn reordered_arrivals_retain_the_same_window() -> Result<(), &'static str> {
    let start = Instant::now();
    let quiet = Duration::from_millis(30);
    let recent = start + Duration::from_millis(115);
    let forward = PrefillWindow::new(quiet, [start, recent].into_iter()).ok_or("two requests")?;
    let reverse = PrefillWindow::new(quiet, [recent, start].into_iter()).ok_or("two requests")?;
    assert_eq!(forward.remaining(recent), reverse.remaining(recent));
    Ok(())
}

#[test]
fn disabled_collection_and_empty_queues_do_not_wait() -> Result<(), &'static str> {
    let now = Instant::now();
    assert!(PrefillWindow::new(Duration::from_secs(1), std::iter::empty()).is_none());
    let window = PrefillWindow::new(Duration::ZERO, [now].into_iter()).ok_or("one request")?;
    assert_eq!(window.remaining(now), Duration::ZERO);
    Ok(())
}

#[test]
fn long_arrival_widens_from_original_timestamps() -> Result<(), &'static str> {
    let start = Instant::now();
    let next = start + Duration::from_millis(2);
    let mut window =
        PrefillWindow::new(Duration::from_millis(3), [start].into_iter()).ok_or("request")?;
    window.arrived(next);
    window.widen(Duration::from_millis(30));
    assert_eq!(window.remaining(next), Duration::from_millis(30));
    assert_eq!(window.remaining(start + Duration::from_millis(32)), Duration::ZERO);
    window.widen(Duration::from_millis(3));
    assert_eq!(window.remaining(next), Duration::from_millis(30));
    Ok(())
}

#[test]
fn widening_does_not_restart_an_aged_queue() -> Result<(), &'static str> {
    let start = Instant::now();
    let mut window =
        PrefillWindow::new(Duration::from_millis(3), [start].into_iter()).ok_or("request")?;
    window.widen(Duration::from_millis(30));
    assert_eq!(window.remaining(start + Duration::from_millis(40)), Duration::ZERO);
    window.arrived(start + Duration::from_millis(115));
    assert_eq!(window.remaining(start + Duration::from_millis(115)), Duration::from_millis(5));
    Ok(())
}
