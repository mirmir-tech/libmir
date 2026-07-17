use super::*;

#[test]
fn separates_think_tokens() {
    let mut normalizer = normalizer(Markers {
        reasoning: vec![1],
        content: vec![2],
        ..Markers::default()
    });
    assert!(normalizer.push(1, String::new()).is_none());
    assert_eq!(
        normalizer.push(10, "draft".into()).map(|token| token.channel),
        Some(GenerationChannel::Reasoning)
    );
    assert!(normalizer.push(2, String::new()).is_none());
    assert_eq!(
        normalizer.push(11, "answer".into()).map(|token| token.channel),
        Some(GenerationChannel::Content)
    );
}

#[test]
fn parses_harmony_channel_header() {
    let mut normalizer = normalizer(Markers {
        channel: vec![3],
        channel_body: vec![4],
        ..Markers::default()
    });
    assert!(normalizer.push(3, String::new()).is_none());
    assert!(normalizer.push(12, "analysis".into()).is_none());
    assert!(normalizer.push(4, String::new()).is_none());
    assert_eq!(
        normalizer.push(13, "reason".into()).map(|token| token.channel),
        Some(GenerationChannel::Reasoning)
    );
}

#[test]
fn parses_newline_terminated_thought_channel() {
    let mut normalizer = normalizer(Markers { channel: vec![3], ..Markers::default() });
    assert!(normalizer.push(3, String::new()).is_none());
    let token = normalizer.push(14, "thought\nwork".into());
    assert_eq!(token.as_ref().map(|token| token.channel), Some(GenerationChannel::Reasoning));
    assert_eq!(token.map(|token| token.text), Some("work".to_owned()));
}

#[test]
fn detects_reasoning_opened_by_prompt() {
    assert!(prompt_requests_reasoning("assistant\n<think>\n"));
    assert!(prompt_requests_reasoning("<|channel>thought\n"));
    assert!(prompt_requests_reasoning("<|channel|>analysis<|message|>"));
    assert!(!prompt_requests_reasoning("<think>x</think>\nanswer"));
}

fn normalizer(markers: Markers) -> OutputNormalizer {
    OutputNormalizer { markers, state: State::Content }
}
