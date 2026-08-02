use foundation::protocol::{ChatCompletionRequest, ChatMessage};
use models::{chat::ChatTemplate, layout::ModelLayout, tokenizer::TextTokenizer};

use super::GENERATED;
use crate::Result;

pub fn prompts(layout: &ModelLayout) -> Result<Vec<Vec<u32>>> {
    let template = ChatTemplate::from_layout(layout)?;
    let tokenizer = TextTokenizer::from_layout(layout)?;
    [
        "Hello. Briefly introduce yourself and explain what you can help with.",
        "Explain mixture-of-experts routing, load balancing, and inference trade-offs.",
        "Write a safe Rust function that parses a port number and explain its error handling.",
    ]
    .into_iter()
    .map(|content| {
        let prompt = template.render(&request(content))?;
        Ok(tokenizer
            .encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?
            .token_ids)
    })
    .collect()
}

fn request(content: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "projection-gate".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        max_tokens: Some(GENERATED),
        min_tokens: None,
        ignore_eos: None,
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: Some(0),
    }
}
