#![allow(clippy::print_stdout)]

use std::{
    env,
    io::{self, Write},
    path::PathBuf,
};

use libmir::{
    ChatCompletionRequest, ChatMessage, Error, GenerationOverrides, Library, RuntimeConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let library = Library::new(RuntimeConfig::default());
    let model = library.load(model_path()?, GenerationOverrides::default(), &mut |_| {})?;
    let request = request(&model.handle().id);
    let mut stdout = io::stdout().lock();
    let mut stream_error = None;
    let output = model.generate(&request, &mut |_| {}, &mut |token| {
        if stream_error.is_none()
            && let Err(error) = write!(stdout, "{}", token.text).and_then(|()| stdout.flush())
        {
            stream_error = Some(error);
        }
    })?;
    if let Some(error) = stream_error {
        return Err(error.into());
    }
    writeln!(stdout, "\nfinish_reason={}", output.finish_reason)?;
    Ok(())
}

fn request(model: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
            reasoning_content: None,
        }],
        stream: true,
        max_tokens: Some(256),
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: None,
    }
}

fn model_path() -> Result<PathBuf, Error> {
    env::args_os()
        .nth(1)
        .or_else(|| env::var_os("MODEL"))
        .map(PathBuf::from)
        .ok_or(Error::MissingEnvironment("MODEL or the first argument"))
}
