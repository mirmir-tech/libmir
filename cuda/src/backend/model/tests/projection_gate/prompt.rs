use foundation::conversation::{Conversation, Message};
use models::{chat::ChatTemplate, layout::ModelLayout, tokenizer::TextTokenizer};

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
