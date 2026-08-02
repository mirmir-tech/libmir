use std::{
    path::PathBuf,
    sync::mpsc::{Receiver, SyncSender, sync_channel},
    thread,
    time::{Duration, Instant},
};

use libmir::{
    ChatCompletionRequest, ChatMessage, GenerationOverrides, IMAGE_PLACEHOLDER, Library, Result,
    RuntimeConfig, SamplingLogits,
};

const BLACK_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 4, 0,
    0, 0, 181, 28, 12, 2, 0, 0, 0, 11, 73, 68, 65, 84, 120, 218, 99, 100, 248, 15, 0, 1, 5, 1, 1,
    39, 24, 227, 102, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const BASELINE_TOKENS: usize = 4;
const CONTENDED_TOKENS: usize = 8;

#[test]
#[ignore = "loads a real vision checkpoint; set MODEL and LIBMIR_VISION_CONTENTION_REPORT"]
fn records_decode_latency_during_vision_prefill() -> Result<()> {
    let model_path = required_path("MODEL")?;
    let report_path = required_path("LIBMIR_VISION_CONTENTION_REPORT")?;
    let mut config = RuntimeConfig::default();
    config.kv_cache.block_count = 128;
    config.scheduler.decode_batch_wait_us = 0;
    config.scheduler.decode_priority_burst = 2;
    let model =
        Library::new(config).load(model_path, GenerationOverrides::default(), &mut |_event| {})?;
    let prepared = model.prepare_image(&vision_request(&model), BLACK_PNG)?;
    model
        .session()
        .prefill_vision(&prepared, SamplingLogits::None, &mut |_event| {})?;

    let prompt = model.prepare(&text_request(&model))?.tokens.token_ids;
    let mut decoding = model.session();
    let mut token =
        next_token(&decoding.prefill(&prompt, SamplingLogits::None, &mut |_event| {})?)?;
    let mut baseline = Vec::with_capacity(BASELINE_TOKENS);
    for _index in 0..BASELINE_TOKENS {
        token = timed_decode(&mut decoding, token, &mut baseline)?;
    }

    let mut vision = model.session();
    let (ready_tx, ready_rx) = sync_channel(0);
    let (start_tx, start_rx) = sync_channel(0);
    let vision_worker = thread::spawn(move || {
        let mut first = true;
        let mut started = None;
        let output = vision.prefill_vision(&prepared, SamplingLogits::None, &mut |_event| {
            if first {
                let _sent = ready_tx.send(());
                started = start_rx.recv().ok();
                first = false;
            }
        });
        (started.map(|instant: Instant| instant.elapsed()), output)
    });
    receive(&ready_rx)?;
    send(&start_tx, Instant::now())?;
    thread::yield_now();

    let overlapped = !vision_worker.is_finished();
    let mut contended = Vec::with_capacity(CONTENDED_TOKENS);
    while contended.len() < CONTENDED_TOKENS && !vision_worker.is_finished() {
        token = timed_decode(&mut decoding, token, &mut contended)?;
    }
    let Ok((vision_elapsed, vision_output)) = vision_worker.join() else {
        return Err(scheduler_error("vision worker panicked").into());
    };
    let vision_output = vision_output?;
    if !overlapped || contended.is_empty() || vision_output.next_token.is_none() {
        return Err(scheduler_error("vision and decode work did not overlap").into());
    }
    if let Err(error) =
        std::fs::write(&report_path, report(&baseline, &contended, vision_elapsed, token))
    {
        return Err(runtime::RuntimeError::Backend(format!(
            "failed to write contention report {}: {error}",
            report_path.display()
        ))
        .into());
    }
    Ok(())
}

fn timed_decode(
    session: &mut libmir::Session,
    token: u32,
    samples: &mut Vec<Duration>,
) -> Result<u32> {
    let started = Instant::now();
    let output = session.decode(token, SamplingLogits::None)?;
    samples.push(started.elapsed());
    output
        .event
        .token_id
        .ok_or_else(|| runtime::RuntimeError::Scheduler("decode returned no token".into()).into())
}

fn next_token(output: &libmir::PrefillOutput) -> Result<u32> {
    output
        .next_token
        .ok_or_else(|| runtime::RuntimeError::Scheduler("prefill returned no token".into()).into())
}

fn report(
    baseline: &[Duration],
    contended: &[Duration],
    vision: Option<Duration>,
    final_token: u32,
) -> String {
    format!(
        "baseline_ms={:?}\ncontended_ms={:?}\nvision_ms={:?}\ncontended_tokens={}\nfinal_token={}\n",
        millis(baseline),
        millis(contended),
        vision.map(|value| value.as_secs_f64() * 1_000.0),
        contended.len(),
        final_token,
    )
}

fn millis(samples: &[Duration]) -> Vec<f64> {
    samples.iter().map(|value| value.as_secs_f64() * 1_000.0).collect()
}

fn required_path(name: &'static str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or(libmir::Error::MissingEnvironment(name))
}

fn receive(receiver: &Receiver<()>) -> Result<()> {
    if receiver.recv().is_err() {
        return Err(scheduler_error("vision ready signal closed").into());
    }
    Ok(())
}

fn send(sender: &SyncSender<Instant>, started: Instant) -> Result<()> {
    if sender.send(started).is_err() {
        return Err(scheduler_error("vision start signal closed").into());
    }
    Ok(())
}

fn scheduler_error(message: &str) -> runtime::RuntimeError {
    runtime::RuntimeError::Scheduler(message.into())
}

fn text_request(model: &libmir::Model) -> ChatCompletionRequest {
    request(model, "Hi")
}

fn vision_request(model: &libmir::Model) -> ChatCompletionRequest {
    request(model, &format!("{IMAGE_PLACEHOLDER}Describe the image."))
}

fn request(model: &libmir::Model, content: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.handle().id.clone(),
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
        max_tokens: Some(CONTENDED_TOKENS),
        min_tokens: None,
        ignore_eos: None,
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: Some(7),
    }
}
