use foundation::conversation::{
    Conversation, FunctionCall, FunctionDefinition, Message, Tool, ToolCall,
};
use libmir_models::{
    ModelsError, Result,
    chat::ChatTemplate,
    layout::{AttentionLayerType, DecoderConfig, ModelLayout},
    tokenizer::{TextTokenizer, TokenizerKind},
};

#[test]
#[ignore = "requires official Ministral-8B-Instruct-2410; set MINISTRAL_MODEL"]
fn loads_official_ministral_interleaved_contract() -> Result<()> {
    let root = std::env::var_os("MINISTRAL_MODEL")
        .ok_or_else(|| ModelsError::InvalidConfig("missing MINISTRAL_MODEL".into()))?;
    let layout = ModelLayout::inspect(root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let metadata = libmir_models::layout::ModelMetadata::from_layout(&layout)?;
    let prompt =
        ChatTemplate::from_layout(&layout)?.render(&request("What is 17 * 23? Answer briefly."))?;
    let tokenizer = TextTokenizer::from_layout(&layout)?;
    let encoded = tokenizer.encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?;

    assert_eq!(decoder.layer_type(0), AttentionLayerType::Full);
    assert_eq!(decoder.layer_type(1), AttentionLayerType::Sliding);
    assert_eq!(decoder.layer_sliding_window(0), None);
    assert_eq!(decoder.layer_sliding_window(1), Some(32768));
    assert_eq!(decoder.max_position_embeddings, 131_072);
    assert_eq!(metadata.context_len, 131_072);
    assert_eq!(prompt.text, "<s>[INST]What is 17 * 23? Answer briefly.[/INST]");
    assert!(!prompt.add_special_tokens);
    assert_eq!(tokenizer.info().kind, TokenizerKind::TokenizerJson);
    assert_eq!(encoded.token_ids.first(), Some(&1));
    assert!(encoded.token_ids.contains(&3));
    assert!(encoded.token_ids.contains(&4));
    assert_eq!(
        encoded.token_ids,
        [1, 3, 7493, 1395, 1032, 1049, 1055, 1364, 1032, 1050, 1051, 1063, 3450, 27457, 1046, 4]
    );
    assert!(tokenizer.stop_token_ids().contains(&2));
    Ok(())
}

#[test]
#[ignore = "requires official Ministral-8B-Instruct-2410; set MINISTRAL_MODEL"]
fn renders_official_ministral_tool_conversation() -> Result<()> {
    let root = std::env::var_os("MINISTRAL_MODEL")
        .ok_or_else(|| ModelsError::InvalidConfig("missing MINISTRAL_MODEL".into()))?;
    let layout = ModelLayout::inspect(root)?;
    let prompt = ChatTemplate::from_layout(&layout)?.render(&tool_request())?;
    let tokenizer = TextTokenizer::from_layout(&layout)?;
    let encoded = tokenizer.encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?;

    assert!(prompt.text.contains("[AVAILABLE_TOOLS]"));
    assert!(prompt.text.contains("[TOOL_CALLS]"));
    assert!(prompt.text.contains("[TOOL_RESULTS]"));
    for marker in [5, 6, 7, 8, 9] {
        assert!(encoded.token_ids.contains(&marker));
    }
    Ok(())
}

fn tool_request() -> Conversation {
    let mut request = request("Weather in Warsaw?");
    request.tools.push(Tool {
        kind: "function".into(),
        function: FunctionDefinition {
            name: "weather".into(),
            description: Some("Get current weather".into()),
            parameters: serde_json::json!({"type": "object"}),
        },
    });
    request.messages.extend([
        Message {
            role: "assistant".into(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: Some(vec![ToolCall {
                id: "abc123456".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({"city": "Warsaw"}),
                },
            }]),
            tool_call_id: None,
        },
        Message {
            role: "tool".into(),
            content: r#"{"temperature":12}"#.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some("abc123456".into()),
        },
    ]);
    request
}

fn request(content: &str) -> Conversation {
    Conversation {
        messages: vec![Message {
            role: "user".into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: foundation::conversation::ToolChoice::default(),
    }
}
