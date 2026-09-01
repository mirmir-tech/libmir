#![allow(clippy::print_stdout)]

use std::{fs, time::Instant};

use libmir::{Conversation, GenerationOverrides, Library, Message, SamplingLogits};

mod prefill_profile;

use prefill_profile::Config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init(),
    );
    let config = Config::parse()?;
    let mut runtime = config.runtime();
    runtime.scheduler.max_batch_requests = 1;
    let model =
        Library::new(runtime).load(&config.model, GenerationOverrides::default(), &mut |_| {})?;
    let tokens = if let Some(path) = &config.token_source {
        let text = fs::read_to_string(path)?;
        let source = model.descriptor().tokenizer().encode(&text)?;
        let source = source
            .token_ids
            .get(config.token_offset..)
            .ok_or("prefill profile token offset exceeds the source")?;
        exact_tokens(source, config.prompt_tokens)?
    } else {
        let prepared = model.prepare(&request(&model, &config.prompt))?;
        exact_tokens(&prepared.tokens.token_ids, config.prompt_tokens)?
    };
    config.write_requests(&tokens)?;
    for rotation in 1..=config.warmup_runs {
        let mut warmup = tokens.clone();
        let length = warmup.len();
        warmup.rotate_left(rotation % length);
        model.session().prefill(&warmup, SamplingLogits::None, &mut |_| {})?;
    }
    model.engine().clear_prefix_cache(model.handle())?;
    if config.prefix_tokens > 0 {
        let mut seed = tokens.clone();
        let suffix = &mut seed[config.prefix_tokens..];
        let suffix_len = suffix.len();
        suffix.rotate_left(1 % suffix_len);
        model.session().prefill(&seed, SamplingLogits::None, &mut |_| {})?;
    }
    model.engine().start_profiler_capture()?;
    let started = Instant::now();
    let mut session = model.session();
    session.prefill(&tokens, SamplingLogits::None, &mut |_| {})?;
    let elapsed = started.elapsed();
    model.engine().stop_profiler_capture()?;
    let measured_tokens = f64::from(u32::try_from(tokens.len() - config.prefix_tokens)?);
    println!(
        "backend={} prompt_tokens={} prefix_tokens={} chunk_capacity={} elapsed_ms={:.3} \
         tokens_per_second={:.3}",
        model.handle().backend,
        tokens.len(),
        config.prefix_tokens,
        config.chunk_tokens,
        elapsed.as_secs_f64() * 1_000.0,
        measured_tokens / elapsed.as_secs_f64(),
    );
    Ok(())
}

fn exact_tokens(source: &[u32], count: usize) -> Result<Vec<u32>, &'static str> {
    if source.is_empty() {
        return Err("prefill profile requires non-empty tokens");
    }
    let count = if count == 0 {
        source.len()
    } else {
        count
    };
    Ok(source.iter().copied().cycle().take(count).collect())
}

fn request(_model: &libmir::Model, prompt: &str) -> Conversation {
    Conversation {
        messages: vec![Message {
            role: "user".into(),
            content: prompt.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: libmir::ToolChoice::default(),
    }
}
