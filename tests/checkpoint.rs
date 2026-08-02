use libmir::{
    ChatCompletionRequest, ChatMessage, GenerationOverrides, Library, Result, RuntimeConfig,
    SamplingLogits,
};

#[path = "checkpoint/prefill_batch.rs"]
mod prefill_batch;

#[test]
#[ignore = "loads a real checkpoint; set MODEL"]
fn generates_through_the_public_library_api() -> Result<()> {
    let path = std::env::var_os("MODEL").ok_or(libmir::Error::MissingEnvironment("MODEL"))?;
    let mut config = RuntimeConfig::default();
    config.kv_cache.block_count = 128;
    let library = Library::new(config);
    let model = library.load(path, GenerationOverrides::default(), &mut |_event| {})?;
    let request = ChatCompletionRequest {
        model: model.handle().id.clone(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        max_tokens: Some(8),
        min_tokens: None,
        ignore_eos: None,
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: Some(7),
    };
    let output = model.generate(&request, &mut |_event| {}, &mut |_token| {})?;

    assert_eq!(output.token_ids.len(), 8);
    assert!(output.prompt_tokens > 0);
    Ok(())
}

#[test]
#[ignore = "loads a real checkpoint; set MODEL"]
fn batches_concurrent_public_sessions() -> Result<()> {
    let path = std::env::var_os("MODEL").ok_or(libmir::Error::MissingEnvironment("MODEL"))?;
    let mut config = RuntimeConfig::default();
    config.kv_cache.block_count = 128;
    config.scheduler.max_batch_requests = 2;
    config.scheduler.decode_batch_wait_us = 50_000;
    let model =
        Library::new(config).load(path, GenerationOverrides::default(), &mut |_event| {})?;
    let expected_trace = batch_trace(&model);
    let request = request(&model);
    let prompt = model.prepare(&request)?.tokens.token_ids;
    let longer_prompt = prompt.iter().copied().cycle().take(prompt.len() + 17).collect::<Vec<_>>();
    let mut first = model.session();
    let mut second = model.session();
    let mut first_reference = model.session();
    let mut second_reference = model.session();
    let first_token = first
        .prefill(&prompt, SamplingLogits::None, &mut |_event| {})?
        .next_token
        .ok_or_else(|| runtime::RuntimeError::Backend("prefill returned no token".into()))?;
    let second_token = second
        .prefill(&longer_prompt, SamplingLogits::None, &mut |_event| {})?
        .next_token
        .ok_or_else(|| runtime::RuntimeError::Backend("prefill returned no token".into()))?;
    let first_reference_token = first_reference
        .prefill(&prompt, SamplingLogits::None, &mut |_event| {})?
        .next_token
        .ok_or_else(|| runtime::RuntimeError::Backend("prefill returned no token".into()))?;
    let second_reference_token = second_reference
        .prefill(&longer_prompt, SamplingLogits::None, &mut |_event| {})?
        .next_token
        .ok_or_else(|| runtime::RuntimeError::Backend("prefill returned no token".into()))?;
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let first_barrier = barrier.clone();
    let first = std::thread::spawn(move || {
        first_barrier.wait();
        first.decode(first_token, SamplingLogits::None)
    });
    let second_barrier = barrier.clone();
    let second = std::thread::spawn(move || {
        second_barrier.wait();
        second.decode(second_token, SamplingLogits::None)
    });
    barrier.wait();
    let first = join(first)?;
    let second = join(second)?;
    let first_reference = first_reference.decode(first_reference_token, SamplingLogits::None)?;
    let second_reference = second_reference.decode(second_reference_token, SamplingLogits::None)?;
    assert_eq!(first.event.token_id, first_reference.event.token_id);
    assert_eq!(second.event.token_id, second_reference.event.token_id);
    assert_eq!(first.event.text, expected_trace);
    assert_eq!(second.event.text, expected_trace);
    Ok(())
}

fn batch_trace(model: &libmir::Model) -> &'static str {
    if model.handle().backend == "mlx-native" {
        "metal.decode=packed-device-token-pipeline"
    } else {
        "cuda.decode=batch-device-token-pipeline"
    }
}

#[test]
#[ignore = "loads a real checkpoint; set MODEL"]
fn decode_preempts_chunked_prefill() -> Result<()> {
    let path = std::env::var_os("MODEL").ok_or(libmir::Error::MissingEnvironment("MODEL"))?;
    let mut config = RuntimeConfig::default();
    config.kv_cache.block_count = 128;
    config.scheduler.decode_batch_wait_us = 0;
    config.scheduler.decode_priority_burst = 2;
    let model =
        Library::new(config).load(path, GenerationOverrides::default(), &mut |_event| {})?;
    let prompt = model.prepare(&request(&model))?.tokens.token_ids;
    let mut decoding = model.session();
    let next = decoding
        .prefill(&prompt, SamplingLogits::None, &mut |_event| {})?
        .next_token
        .ok_or_else(|| runtime::RuntimeError::Backend("prefill returned no token".into()))?;

    let long_prompt = prompt.iter().copied().cycle().take(512).collect::<Vec<_>>();
    let mut prefill = model.session();
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
    let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(0);
    let prefill = std::thread::spawn(move || {
        let mut first = true;
        prefill.prefill(&long_prompt, SamplingLogits::None, &mut |_event| {
            if first {
                let _sent = started_tx.send(());
                let _resumed = resume_rx.recv();
                first = false;
            }
        })
    });
    receive(&started_rx)?;
    let decode = std::thread::spawn(move || decoding.decode(next, SamplingLogits::None));
    std::thread::sleep(std::time::Duration::from_millis(5));
    send(&resume_tx)?;

    let decode = join(decode)?;
    assert_eq!(decode.event.text, "cuda.decode=device-token-pipeline");
    join_prefill(prefill)?;
    Ok(())
}

fn request(model: &libmir::Model) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.handle().id.clone(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        max_tokens: Some(8),
        min_tokens: None,
        ignore_eos: None,
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: Some(7),
    }
}

fn join(
    handle: std::thread::JoinHandle<Result<libmir::DecodeOutput>>,
) -> Result<libmir::DecodeOutput> {
    let Ok(output) = handle.join() else {
        return Err(runtime::RuntimeError::Scheduler("decode worker panicked".into()).into());
    };
    output
}

fn join_prefill(
    handle: std::thread::JoinHandle<Result<libmir::PrefillOutput>>,
) -> Result<libmir::PrefillOutput> {
    let Ok(output) = handle.join() else {
        return Err(runtime::RuntimeError::Scheduler("prefill worker panicked".into()).into());
    };
    output
}

fn receive(receiver: &std::sync::mpsc::Receiver<()>) -> Result<()> {
    if receiver.recv().is_err() {
        return Err(runtime::RuntimeError::Scheduler("prefill signal closed".into()).into());
    }
    Ok(())
}

fn send(sender: &std::sync::mpsc::SyncSender<()>) -> Result<()> {
    if sender.send(()).is_err() {
        return Err(runtime::RuntimeError::Scheduler("prefill resume closed".into()).into());
    }
    Ok(())
}
