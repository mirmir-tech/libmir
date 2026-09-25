use foundation::conversation::{FunctionDefinition, Tool};
use serde_json::json;

use super::*;

fn conversation() -> Conversation {
    Conversation {
        tools: vec![Tool {
            kind: "function".into(),
            function: FunctionDefinition {
                name: "emit".into(),
                description: None,
                parameters: json!({"type":"object","properties":{"name":{"type":"string"},"year":{"type":"integer"},"items":{"type":"array"}},"required":["name","year","items"]}),
            },
        }],
        tool_choice: ToolChoice::Function("emit".into()),
        ..Default::default()
    }
}

#[test]
fn normalizes_xml_to_openai_calls_without_losing_argument_types() -> Result<()> {
    let input = "<tool_call>\n<function=emit>\n<parameter=name>\n2023\n</parameter>\n<parameter=year>2023</parameter>\n<parameter=items>[{\"value\":4.5,\"known\":true,\"missing\":null}]</parameter>\n</function>\n</tool_call>";
    let normalized = normalize(input, &conversation())?;
    let calls = ToolCall::parse_mistral(&normalized).map_err(Error::InvalidToolCall)?;
    assert_eq!(
        calls[0].function.arguments,
        json!({"name":"2023","year":2023,"items":[{"value":4.5,"known":true,"missing":null}]})
    );
    Ok(())
}

#[test]
fn supports_json_tools_and_rejects_missing_or_wrong_required_calls() -> Result<()> {
    let request = conversation();
    let body = r#"[{"name":"emit","arguments":{"name":"n","year":2023,"items":[]}}]"#;
    assert!(!normalize(body, &request)?.is_empty());
    assert!(
        !normalize(&format!("<tool_call>{}</tool_call>", &body[1..body.len() - 1]), &request)?
            .is_empty()
    );
    assert!(normalize("", &request).is_err());
    assert!(normalize(&body.replace("emit", "wrong"), &request).is_err());
    assert!(normalize(body, &Conversation { tool_choice: ToolChoice::None, ..request }).is_err());
    Ok(())
}

#[test]
fn never_accepts_truncated_or_ambiguous_xml() {
    for input in [
        "<tool_call><function=emit>",
        "<tool_call><function=emit></function></tool_call>",
        "<tool_call><function=emit><parameter=year>not a number</parameter></function></tool_call>",
        "<tool_call><function=emit><parameter=year>2023</parameter><parameter=year>2024</parameter></function></tool_call>",
        "<tool_call><function=emit><parameter=unknown>x</parameter></function></tool_call>",
    ] {
        assert!(normalize(input, &conversation()).is_err());
    }
}

#[test]
fn malformed_nested_argument_reports_local_context_without_repairing_output() -> Result<()> {
    let input = "<tool_call><function=emit><parameter=name>valid</parameter><parameter=year>2023</parameter><parameter=items>[{\"value\":1 \"other\":2}]</parameter></function></tool_call>";
    let error = normalize(input, &conversation())
        .err()
        .ok_or_else(|| invalid("malformed JSON was accepted"))?;
    let Error::InvalidToolCall(message) = error else {
        return Err(error);
    };
    assert!(message.contains("expected `,` or `}`"));
    assert!(message.contains("untrusted"));
    assert!(message.contains("other"));
    assert!(!message.contains("<parameter=name>"));
    Ok(())
}
