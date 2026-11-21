//! A Han-run tiktoken checkpoint (Moonlight) tokenizes a multilingual
//! corpus to the ids the `tiktoken` package produces, written by Workmir's
//! `benchmarks/tools/mla/tiktoken_oracle.py`, and decodes them back.

use libmir_models::{
    ModelsError, Result,
    layout::ModelLayout,
    tokenizer::{TextTokenizer, TokenizerKind},
};
use serde_json::Value;

#[test]
#[ignore = "requires a Han-run tiktoken checkpoint (Moonlight) and its reference; set LIBMIR_TIKTOKEN_MODEL and LIBMIR_TIKTOKEN_REFERENCE"]
fn tokenizes_like_tiktoken() -> Result<()> {
    let variable = |name: &str| {
        std::env::var_os(name).ok_or_else(|| ModelsError::InvalidConfig(format!("missing {name}")))
    };
    let layout = ModelLayout::inspect(variable("LIBMIR_TIKTOKEN_MODEL")?)?;
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(variable("LIBMIR_TIKTOKEN_REFERENCE")?)?)?;
    let tokenizer = TextTokenizer::from_layout(&layout)?;
    assert_eq!(tokenizer.info().kind, TokenizerKind::TikToken);

    let records = reference["records"].as_array().cloned().unwrap_or_default();
    assert!(!records.is_empty());
    for record in records {
        let text = record["text"].as_str().unwrap_or_default();
        let expected = record["ids"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let actual = tokenizer.encode_with_special_tokens(text, false)?.token_ids;
        assert_eq!(actual, expected, "{text:?}");
        // Decoding skips special tokens, so only plain text round-trips.
        if actual.iter().all(|id| {
            tokenizer
                .token(*id)
                .is_some_and(|token| !token.starts_with('<') && !token.starts_with('['))
        }) {
            assert_eq!(tokenizer.decode(&actual)?, text, "{text:?}");
        }
    }
    Ok(())
}

/// The chat template and tokenizer give the prompt ids the greedy oracle
/// (`benchmarks/tools/mla/greedy_oracle.py --tiktoken`) generated from.
#[test]
#[ignore = "requires a Han-run tiktoken checkpoint (Moonlight) and its greedy reference; set LIBMIR_TIKTOKEN_MODEL and LIBMIR_TIKTOKEN_GREEDY"]
fn renders_the_greedy_oracle_prompts() -> Result<()> {
    let variable = |name: &str| {
        std::env::var_os(name).ok_or_else(|| ModelsError::InvalidConfig(format!("missing {name}")))
    };
    let layout = ModelLayout::inspect(variable("LIBMIR_TIKTOKEN_MODEL")?)?;
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(variable("LIBMIR_TIKTOKEN_GREEDY")?)?)?;
    let template = libmir_models::chat::ChatTemplate::from_layout(&layout)?;
    let tokenizer = TextTokenizer::from_layout(&layout)?;
    for record in reference["records"].as_array().cloned().unwrap_or_default() {
        let content = record["prompt"].as_str().unwrap_or_default().to_owned();
        let prompt = template.render(&foundation::conversation::Conversation {
            messages: vec![foundation::conversation::Message {
                role: "user".into(),
                content,
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Vec::new(),
            tool_choice: foundation::conversation::ToolChoice::default(),
        })?;
        let expected = record["prompt_ids"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let actual = tokenizer
            .encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?
            .token_ids;
        assert_eq!(actual, expected, "{:?}", prompt.text);
    }
    Ok(())
}
