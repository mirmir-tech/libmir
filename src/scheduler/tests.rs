use std::time::Duration;

use super::{collection_target, decode_admission_wait};

#[test]
fn decode_admission_uses_the_configured_wait_without_hidden_amplification() {
    assert_eq!(decode_admission_wait(200), Duration::from_micros(200));
}

#[test]
fn admission_window_grows_until_the_batch_limit() {
    assert_eq!(collection_target(1, true, 16), 2);
    assert_eq!(collection_target(10, true, 16), 11);
    assert_eq!(collection_target(16, true, 16), 16);
    assert_eq!(collection_target(10, false, 16), 10);
}
