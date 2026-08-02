use std::io::{self, Write};

use libmir::{Library, MemorySnapshot, Model, SamplingLogits};

use super::{
    TestResult,
    fixture::{Family, Reference, active_target, require, validation_error},
    logits::validate_logits,
};

pub fn validate(
    library: &Library,
    model: &Model,
    reference: &Reference,
    baseline: &MemorySnapshot,
    loaded: &MemorySnapshot,
) -> TestResult<()> {
    let gate = reference
        .gate(&active_target())
        .ok_or_else(|| validation_error("reference has no resource gate for active backend"))?;
    require(
        active_delta(baseline, loaded) <= gate.max_load_active_bytes,
        format!(
            "model load active-memory delta {} exceeds {} bytes",
            active_delta(baseline, loaded),
            gate.max_load_active_bytes
        ),
    )?;
    let mut session = model.session();
    let prefill =
        session.prefill(&reference.prompt_tokens, SamplingLogits::Full, &mut |_event| {})?;
    let first = validate_logits(&prefill, reference)?;
    let expected_tokens = reference.tokens(&active_target());

    let mut generated = vec![first];
    let mut token = first;
    for _ in expected_tokens.iter().skip(1) {
        let output = session.decode(token, SamplingLogits::None)?;
        token = output
            .event
            .token_id
            .ok_or_else(|| validation_error("decode did not return a token"))?;
        generated.push(token);
    }
    require(
        gate.allows_generation(&generated, expected_tokens),
        format!("greedy generation differs: actual={generated:?}, expected={expected_tokens:?}"),
    )?;
    let after_decode = library.memory_snapshot()?;
    require(
        active_delta(baseline, &after_decode) <= gate.max_decode_active_bytes,
        format!(
            "decode active-memory delta {} exceeds {} bytes",
            active_delta(baseline, &after_decode),
            gate.max_decode_active_bytes
        ),
    )?;
    drop(session);
    if reference.affine.is_some()
        || reference.packed_int8.is_some()
        || reference.packed_int4.is_some()
        || reference.awq.is_some()
        || reference.gptq.is_some()
        || reference.float8.is_some()
        || reference.mxfp4.is_some()
        || reference.mxfp8.is_some()
        || reference.nvfp4.is_some()
        || reference.bitsandbytes_4bit.is_some()
    {
        validate_packed_prefill(model, reference)?;
    }
    validate_device_batch(model, reference)?;
    report(
        active_delta(baseline, loaded),
        active_delta(baseline, &after_decode),
        reference.prompt_tokens.len(),
        generated.len(),
    )
}

fn report(
    load_bytes: u64,
    decode_bytes: u64,
    prompt_tokens: usize,
    generated_tokens: usize,
) -> TestResult<()> {
    writeln!(
        io::stderr().lock(),
        "dense checkpoint metrics: load_bytes={load_bytes} decode_bytes={decode_bytes} \
         prompt_tokens={prompt_tokens} generated_tokens={generated_tokens}"
    )?;
    Ok(())
}

fn validate_packed_prefill(model: &Model, reference: &Reference) -> TestResult<()> {
    model.engine().clear_prefix_cache(model.handle())?;
    let mut first = model.session();
    let mut second = model.session();
    let first_prompt = reference.prompt_tokens.clone();
    let second_prompt = first_prompt.clone();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let first_barrier = barrier.clone();
    let first = std::thread::spawn(move || {
        first_barrier.wait();
        first.prefill(&first_prompt, SamplingLogits::None, &mut |_event| {})
    });
    let second_barrier = barrier.clone();
    let second = std::thread::spawn(move || {
        second_barrier.wait();
        second.prefill(&second_prompt, SamplingLogits::None, &mut |_event| {})
    });
    barrier.wait();
    let Ok(first) = first.join() else {
        return Err(validation_error("first packed-prefill worker panicked"));
    };
    let Ok(second) = second.join() else {
        return Err(validation_error("second packed-prefill worker panicked"));
    };
    let expected = Some(reference.tokens(&active_target())[0]);
    require(first?.next_token == expected, "first packed prefill selected the wrong token")?;
    require(second?.next_token == expected, "second packed prefill selected the wrong token")
}

fn validate_device_batch(model: &Model, reference: &Reference) -> TestResult<()> {
    model.engine().clear_prefix_cache(model.handle())?;
    let mut first = model.session();
    let mut second = model.session();
    let first_token = first
        .prefill(&reference.prompt_tokens, SamplingLogits::None, &mut |_event| {})?
        .next_token
        .ok_or_else(|| validation_error("first batch prefill did not return a token"))?;
    let second_token = second
        .prefill(&reference.prompt_tokens, SamplingLogits::None, &mut |_event| {})?
        .next_token
        .ok_or_else(|| validation_error("second batch prefill did not return a token"))?;
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
    let Ok(first) = first.join() else {
        return Err(validation_error("first batch worker panicked"));
    };
    let first = first?;
    let Ok(second) = second.join() else {
        return Err(validation_error("second batch worker panicked"));
    };
    let second = second?;
    let expected_token = reference.tokens(&active_target())[1];
    require(
        first.event.token_id == Some(expected_token),
        format!(
            "first batched token differs: actual={:?}, expected={expected_token}",
            first.event.token_id
        ),
    )?;
    require(
        second.event.token_id == Some(expected_token),
        format!(
            "second batched token differs: actual={:?}, expected={expected_token}",
            second.event.token_id
        ),
    )?;
    let expected_trace = match (model.handle().backend.as_str(), reference.family) {
        ("mlx-native", _) => "metal.decode=packed-device-token-pipeline",
        ("cuda-native", Family::SharedRouted | Family::ClampedRouted) => {
            "cuda.decode=device-token-pipeline"
        },
        _ => "cuda.decode=batch-device-token-pipeline",
    };
    require(
        first.event.text == expected_trace,
        format!(
            "first decode trace differs: actual={:?}, expected={expected_trace:?}",
            first.event.text
        ),
    )?;
    require(
        second.event.text == expected_trace,
        format!(
            "second decode trace differs: actual={:?}, expected={expected_trace:?}",
            second.event.text
        ),
    )
}

fn active_delta(baseline: &MemorySnapshot, current: &MemorySnapshot) -> u64 {
    current.active_bytes.saturating_sub(baseline.active_bytes)
}
