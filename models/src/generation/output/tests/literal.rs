use super::*;

#[test]
fn constrained_json_preserves_all_protocol_markers() {
    let markers = Markers {
        reasoning: vec![1],
        content: vec![2],
        turn_start: vec![3],
        channel: vec![4],
        channel_body: vec![5],
        channel_end: vec![6],
        tool_calls: vec![7],
        xml_tool_start: vec![8],
        xml_tool_end: vec![9],
    };
    let mut n = normalizer(markers);
    let mut text = String::new();
    for (id, piece) in [
        (10, "{\"text\":\""),
        (1, "<think>"),
        (2, "</think>"),
        (3, "<|start|>"),
        (4, "<|channel|>"),
        (5, "<|message|>"),
        (6, "<|end|>"),
        (7, "[TOOL_CALLS]"),
        (8, "<tool_call>"),
        (9, "</tool_call>"),
        (11, "\"}"),
    ] {
        let delta = n.push_literal(id, piece.into(), Some(GenerationChannel::Content));
        assert_eq!(delta.as_ref().map(|t| t.channel), Some(GenerationChannel::Content));
        assert_eq!(delta.as_ref().map(|t| t.text.as_str()), Some(piece));
        if let Some(delta) = delta {
            text.push_str(&delta.text);
        }
    }
    assert!(serde_json::from_str::<serde_json::Value>(&text).is_ok());
}

#[test]
fn reasoning_boundaries_and_buffered_ids_preserve_stream_channels() {
    let mut n = normalizer(Markers::default());
    assert!(n.push_literal(1, "<think>".into(), None).is_none());
    let reasoning = n.push_literal(2, "draft".into(), Some(GenerationChannel::Reasoning));
    assert_eq!(reasoning.as_ref().map(|t| t.channel), Some(GenerationChannel::Reasoning));
    assert_eq!(reasoning.map(|t| t.preceding_ids), Some(vec![1]));
    assert!(n.push_literal(3, "</think>".into(), None).is_none());
    assert!(n.push_literal(4, String::new(), Some(GenerationChannel::Content)).is_none());
    let tail = n.finish("é".into());
    assert_eq!(tail.as_ref().map(|t| t.channel), Some(GenerationChannel::Content));
    assert_eq!(tail.map(|t| (t.id, t.preceding_ids, t.text)), Some((4, vec![3], "é".into())));
}
