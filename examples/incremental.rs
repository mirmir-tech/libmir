#![allow(clippy::print_stdout)]

use std::{
    env,
    io::{self, Write},
    path::PathBuf,
};

use libmir::{
    Conversation, Error, GenerationOverrides, Library, Message, RuntimeConfig, RuntimeError,
    SamplingLogits,
};

const TOKEN_LIMIT: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Library::new(RuntimeConfig::default()).load(
        model_path()?,
        GenerationOverrides::default(),
        &mut |_| {},
    )?;
    let prepared = model.prepare(&request(&model.handle().id))?;
    let mut session = model.session();
    let prefill = session.prefill(&prepared.tokens.token_ids, SamplingLogits::None, &mut |_| {})?;
    let mut next = required_token(prefill.next_token)?;
    let mut stdout = io::stdout().lock();

    for _ in 0..TOKEN_LIMIT {
        write!(stdout, "{}", model.descriptor().tokenizer().decode(&[next])?)?;
        stdout.flush()?;
        if model.descriptor().tokenizer().stop_token_ids().contains(&next) {
            break;
        }
        next = required_token(session.decode(next, SamplingLogits::None)?.event.token_id)?;
    }
    writeln!(stdout, "\ncache={:?}", session.cache_stats())?;
    Ok(())
}

fn required_token(token: Option<u32>) -> Result<u32, RuntimeError> {
    token.ok_or_else(|| RuntimeError::Backend("device sampling returned no token".into()))
}

fn request(_model: &str) -> Conversation {
    Conversation {
        messages: vec![Message {
            role: "user".into(),
            content: "Hello".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: libmir::ToolChoice::default(),
    }
}

fn model_path() -> Result<PathBuf, Error> {
    env::args_os()
        .nth(1)
        .or_else(|| env::var_os("MODEL"))
        .map(PathBuf::from)
        .ok_or(Error::MissingEnvironment("MODEL or the first argument"))
}
