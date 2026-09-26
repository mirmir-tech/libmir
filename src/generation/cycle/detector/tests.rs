use super::*;

#[test]
fn repeated_evidence_with_new_analysis_does_not_end_reasoning() {
    let mut detector = CycleDetector::reasoning_exit(64);
    let quote = (10..22).collect::<Vec<_>>();
    let mut tokens = (1000..1064).collect::<Vec<_>>();
    for index in 0..5 {
        tokens.extend(&quote);
        tokens.extend((2000 + index * 32)..(2032 + index * 32));
    }
    for end in 1..=tokens.len() {
        assert!(detector.observe(&tokens[..end]).is_none(), "false exit at {end}");
    }
}

#[test]
fn reasoning_requires_three_consecutive_cycles_not_a_pair() {
    let mut detector = CycleDetector::reasoning_exit(64);
    let mut tokens = (1000..1064).collect::<Vec<_>>();
    tokens.extend([7, 11, 13, 17].repeat(2));
    assert!(detector.observe(&tokens).is_none());
    tokens.extend([7, 11, 13, 17]);
    assert!(matches!(
        detector.observe(&tokens),
        Some(CycleDetection { span: 4, kind: CycleKind::Consecutive })
    ));
}

#[test]
fn unconstrained_recovery_still_detects_nonconsecutive_phrase_cycles() {
    let mut detector = CycleDetector::default();
    let quote = (10..22).collect::<Vec<_>>();
    let mut tokens = (1000..1200).collect::<Vec<_>>();
    for index in 0..3 {
        tokens.extend(&quote);
        tokens.extend((2000 + index * 32)..(2032 + index * 32));
    }
    tokens.extend(quote);
    assert!(matches!(
        detector.observe(&tokens),
        Some(CycleDetection { kind: CycleKind::RecurringPhrase, .. })
    ));
}
