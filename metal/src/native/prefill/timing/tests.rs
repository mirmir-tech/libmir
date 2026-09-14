use super::*;

#[test]
fn separates_execution_from_queue_and_interleaving_without_sleep() {
    let start = Instant::now();
    let mut clock = PrefillTiming::new(start);
    for ms in [3, 12, 7, 2] {
        clock.record(Duration::from_millis(ms));
    }
    let (elapsed, timing) = clock.finish_at(start + Duration::from_millis(100));
    assert_eq!(elapsed, Duration::from_millis(100));
    assert_eq!(timing.backend_execution, Duration::from_millis(24));
    assert_eq!(timing.backend_wait, Duration::from_millis(76));
    assert_eq!(timing.cache_prepare, Duration::ZERO);
    assert_eq!(timing.scheduler_queue, Duration::ZERO);
}

#[test]
fn shared_step_belongs_to_each_row_but_scalar_completion_does_not() {
    let start = Instant::now();
    let mut first = PrefillTiming::new(start);
    let mut second = PrefillTiming::new(start);
    first.record(Duration::from_millis(10));
    second.record(Duration::from_millis(10));
    first.record(Duration::from_millis(4));
    second.record(Duration::from_millis(2));
    let (_, first) = first.finish_at(start + Duration::from_millis(20));
    let (_, second) = second.finish_at(start + Duration::from_millis(30));
    assert_eq!(first.backend_execution, Duration::from_millis(14));
    assert_eq!(first.backend_wait, Duration::from_millis(6));
    assert_eq!(second.backend_execution, Duration::from_millis(12));
    assert_eq!(second.backend_wait, Duration::from_millis(18));
}
