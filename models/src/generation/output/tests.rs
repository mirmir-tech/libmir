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
fn separates_mistral_tool_calls() {
    let mut normalizer = normalizer(Markers {
        tool_calls: vec![9],
        ..Markers::default()
    });
    assert!(normalizer.push(9, String::new()).is_none());
    let token = normalizer.push(10, r#"[{"name":"weather"}]"#.into());
    assert_eq!(token.as_ref().map(|token| token.channel), Some(GenerationChannel::ToolCalls));
    assert_eq!(token.map(|token| token.text), Some(r#"[{"name":"weather"}]"#.to_owned()));
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
fn suppresses_harmony_role_before_final_channel() {
    let mut normalizer = normalizer(Markers {
        turn_start: vec![2],
        channel: vec![3],
        channel_body: vec![4],
        ..Markers::default()
    });
    assert!(normalizer.push(2, String::new()).is_none());
    assert!(normalizer.push(11, "assistant".into()).is_none());
    assert!(normalizer.push(3, String::new()).is_none());
    assert!(normalizer.push(12, "final".into()).is_none());
    assert!(normalizer.push(4, String::new()).is_none());
    let token = normalizer.push(13, "answer".into());
    assert_eq!(token.as_ref().map(|token| token.channel), Some(GenerationChannel::Content));
    assert_eq!(token.map(|token| token.text), Some("answer".to_owned()));
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
    OutputNormalizer {
        markers,
        state: State::Content,
        pending_ids: Vec::new(),
    }
}

#[test]
fn retains_ids_buffered_before_visible_text() {
    let mut normalizer = normalizer(Markers::default());
    assert!(normalizer.push(7, String::new()).is_none());
    let token = normalizer.push(8, "x".into());
    assert_eq!(token.as_ref().map(|token| token.preceding_ids.as_slice()), Some([7].as_slice()));
    assert_eq!(token.map(|token| token.id), Some(8));
}

#[test]
fn finish_attaches_released_text_to_withheld_ids() {
    let mut normalizer = normalizer(Markers::default());
    assert!(normalizer.push(7, String::new()).is_none());
    assert!(normalizer.push(8, String::new()).is_none());
    let token = normalizer.finish("\u{fffd}".into());
    assert_eq!(token.as_ref().map(|token| token.preceding_ids.as_slice()), Some([7].as_slice()));
    assert_eq!(token.as_ref().map(|token| token.id), Some(8));
    assert_eq!(token.map(|token| token.text), Some("\u{fffd}".into()));
    assert!(normalizer.finish(String::new()).is_none());
}

#[test]
fn routes_xml_markers_and_body_outside_visible_content() {
    let mut normalizer = normalizer(Markers {
        xml_tool_start: vec![20],
        xml_tool_end: vec![21],
        ..Default::default()
    });
    for (id, text) in
        [(20, "<tool_call>"), (22, "<function=emit>"), (23, "</function>"), (21, "</tool_call>")]
    {
        let token = normalizer.push(id, text.into());
        assert_eq!(token.as_ref().map(|token| token.channel), Some(GenerationChannel::ToolCalls));
        assert_eq!(token.map(|token| token.text), Some(text.into()));
    }
    assert_eq!(
        normalizer.push(24, "after".into()).map(|token| token.channel),
        Some(GenerationChannel::Content)
    );
}

#[test]
fn tool_markup_inside_reasoning_is_not_executed() {
    let mut normalizer = normalizer(Markers {
        reasoning: vec![1],
        content: vec![2],
        xml_tool_start: vec![20],
        xml_tool_end: vec![21],
        ..Default::default()
    });
    assert!(normalizer.push(1, String::new()).is_none());
    for (id, text) in [(20, "<tool_call>"), (22, "example"), (21, "</tool_call>")] {
        assert_eq!(
            normalizer.push(id, text.into()).map(|token| token.channel),
            Some(GenerationChannel::Reasoning)
        );
    }
    assert!(normalizer.push(2, String::new()).is_none());
    assert_eq!(
        normalizer.push(20, "<tool_call>".into()).map(|token| token.channel),
        Some(GenerationChannel::ToolCalls)
    );
}

#[test]
fn literal_xml_without_tools_stays_visible_text() {
    let mut normalizer = normalizer(Markers {
        xml_tool_start: vec![20],
        xml_tool_end: vec![21],
        ..Default::default()
    })
    .without_tool_protocol();
    for (id, text) in [(20, "<tool_call>"), (22, "example"), (21, "</tool_call>")] {
        let token = normalizer.push(id, text.into());
        assert_eq!(token.as_ref().map(|token| token.channel), Some(GenerationChannel::Content));
        assert_eq!(token.map(|token| token.text), Some(text.into()));
    }
}
