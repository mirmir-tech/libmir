use foundation::protocol::{ChatCompletionRequest, ChatMessage};
use libmir_models::{
    ModelsError, Result,
    chat::ChatTemplate,
    layout::ModelLayout,
    tokenizer::{TextTokenizer, TokenizerKind},
};

#[test]
#[ignore = "requires official Mistral-7B-Instruct-v0.3; set MISTRAL_MODEL"]
fn renders_and_tokenizes_official_mistral_v3_prompt() -> Result<()> {
    let root = std::env::var_os("MISTRAL_MODEL")
        .ok_or_else(|| ModelsError::InvalidConfig("missing MISTRAL_MODEL".into()))?;
    let layout = ModelLayout::inspect(root)?;
    let prompt = ChatTemplate::from_layout(&layout)?.render(&ChatCompletionRequest {
        model: "mistral".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        max_tokens: None,
        min_tokens: None,
        ignore_eos: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        seed: None,
    })?;
    let tokenizer = TextTokenizer::from_layout(&layout)?;
    let encoded = tokenizer.encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?;

    assert_eq!(prompt.text, "<s>[INST] Hello[/INST]");
    assert!(!prompt.add_special_tokens);
    assert_eq!(tokenizer.info().kind, TokenizerKind::TokenizerJson);
    assert_eq!(encoded.token_ids.first(), Some(&1));
    assert!(tokenizer.stop_token_ids().contains(&2));

    let sentence = tokenizer.encode_with_special_tokens("The product is 391.", false)?;
    let expected = tokenizer.decode(&sentence.token_ids)?;
    let mut actual = String::new();
    let mut decoder = tokenizer.decoder();
    for token_id in sentence.token_ids {
        if let Some(piece) = decoder.step(token_id)? {
            actual.push_str(&piece);
        }
    }
    assert_eq!(actual, expected);
    Ok(())
}
