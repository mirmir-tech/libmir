use super::{Duration, PrefillAdmissionPolicy, collection_wait};

#[test]
fn only_idle_short_queues_receive_the_short_window() {
    let configured = Duration::from_millis(30);
    for lengths in [vec![1], vec![128], vec![128; 5]] {
        assert_eq!(
            collection_wait(
                PrefillAdmissionPolicy::ShortPrompt,
                configured,
                lengths.into_iter(),
                false
            ),
            Duration::from_millis(3)
        );
    }
    for lengths in [vec![129], vec![8192], vec![128, 8192], vec![8192, 128], vec![0]] {
        assert_eq!(
            collection_wait(
                PrefillAdmissionPolicy::ShortPrompt,
                configured,
                lengths.into_iter(),
                false
            ),
            configured
        );
    }
    assert_eq!(
        collection_wait(PrefillAdmissionPolicy::ShortPrompt, configured, [128].into_iter(), true),
        configured
    );
    assert_eq!(
        collection_wait(PrefillAdmissionPolicy::Uniform, configured, [128].into_iter(), false),
        configured
    );
}

#[test]
fn explicit_short_or_disabled_waits_are_preserved() {
    for configured in [Duration::ZERO, Duration::from_micros(600)] {
        assert_eq!(
            collection_wait(
                PrefillAdmissionPolicy::ShortPrompt,
                configured,
                [128].into_iter(),
                false
            ),
            configured
        );
    }
}
