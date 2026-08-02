use std::{
    collections::{HashSet, VecDeque},
    time::Duration,
};

use super::{State, collection_target};

#[test]
fn successful_batch_retains_a_bounded_refill_hint() {
    let mut state = State {
        waiting: VecDeque::new(),
        active: HashSet::new(),
        running: false,
        refill_steps: 0,
    };
    assert_eq!(state.refill_wait(200), Duration::from_micros(200));
    state.observe(4);
    assert_eq!(state.refill_wait(200), Duration::from_millis(5));
    for _ in 0..64 {
        state.observe(1);
    }
    assert_eq!(state.refill_wait(200), Duration::from_micros(200));
}

#[test]
fn admission_window_grows_until_the_batch_limit() {
    assert_eq!(collection_target(1, true, 16), 2);
    assert_eq!(collection_target(10, true, 16), 11);
    assert_eq!(collection_target(16, true, 16), 16);
    assert_eq!(collection_target(10, false, 16), 10);
}
