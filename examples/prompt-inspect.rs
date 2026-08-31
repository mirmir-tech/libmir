#![allow(clippy::print_stdout)]

use std::{
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use libmir::{Conversation, Error, GenerationOverrides, Message, ModelDescriptor};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    key: String,
    prompt: String,
}

#[derive(Serialize)]
struct PromptRecord {
    key: String,
    text: String,
    token_ids: Vec<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = argument(1, "model path")?;
    let cases_path = argument(2, "cases JSONL path")?;
    let descriptor = ModelDescriptor::inspect(model_path, GenerationOverrides::default())?;
    for line in BufReader::new(File::open(cases_path)?).lines() {
        let case: Case = serde_json::from_str(&line?)?;
        let prepared = descriptor.prepare(&Conversation {
            messages: vec![Message {
                role: "user".into(),
                content: case.prompt,
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Vec::new(),
            tool_choice: libmir::ToolChoice::default(),
        })?;
        println!(
            "{}",
            serde_json::to_string(&PromptRecord {
                key: case.key,
                text: prepared.prompt.text,
                token_ids: prepared.tokens.token_ids,
            })?
        );
    }
    Ok(())
}

fn argument(index: usize, name: &'static str) -> Result<PathBuf, Error> {
    env::args_os()
        .nth(index)
        .map(PathBuf::from)
        .ok_or(Error::MissingEnvironment(name))
}
