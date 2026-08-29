#![allow(clippy::print_stdout)]

use std::time::{Duration, Instant};

use libmir::{
    Conversation, DecodeTimings, GenerationOverrides, Library, Message, RuntimeConfig,
    RuntimeError, SamplingLogits,
};

mod decode_profile;

use decode_profile::Config;

struct Sample {
    end_to_end: Duration,
    timings: DecodeTimings,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init(),
    );
    let config = Config::parse()?;
    let mut runtime = RuntimeConfig::default();
    runtime.scheduler.max_batch_requests = 1;
    runtime.scheduler.decode_batch_wait_us = 0;
    config.configure(&mut runtime);
    let model =
        Library::new(runtime).load(&config.model, GenerationOverrides::default(), &mut |_| {})?;
    let base = model.prepare(&request(&model))?.tokens.token_ids;
    let prompt = base.iter().copied().cycle().take(config.prompt_tokens).collect::<Vec<_>>();
    fill_prefix_cache(&model, &prompt, config.prefix_fill_sessions)?;
    if config.clear_allocator_after_fill {
        model.engine().clear_memory_cache()?;
    }
    std::thread::sleep(Duration::from_secs(config.prefix_fill_cooldown_seconds));
    let mut session = model.session();
    let output = session.prefill(&prompt, SamplingLogits::None, &mut |_| {})?;
    let mut token = required_token(output.next_token)?;
    for _ in 0..config.warmup_steps {
        token = required_token(session.decode(token, SamplingLogits::None)?.event.token_id)?;
    }
    model.engine().set_profile_decode(true)?;
    let mut samples = Vec::with_capacity(config.measured_steps);
    for _ in 0..config.measured_steps {
        let started = Instant::now();
        let output = session.decode(token, SamplingLogits::None)?;
        let end_to_end = started.elapsed();
        token = required_token(output.event.token_id)?;
        let timings = output
            .timings
            .ok_or("backend did not return enabled decode profiling timings")?;
        samples.push(Sample { end_to_end, timings });
    }
    model.engine().set_profile_decode(false)?;
    report(model.handle().backend.as_str(), &config, &samples);
    Ok(())
}

fn report(backend: &str, config: &Config, samples: &[Sample]) {
    let end_to_end = samples.iter().map(|sample| sample.end_to_end).collect::<Vec<_>>();
    let queue = samples.iter().map(|sample| sample.timings.scheduler_queue);
    let wait = samples.iter().map(|sample| sample.timings.backend_wait);
    let backend_wall = samples.iter().map(|sample| sample.timings.backend_execution);
    let device = samples.iter().filter_map(|sample| sample.timings.device_execution);
    let backend_minus_device = samples.iter().filter_map(|sample| {
        sample
            .timings
            .device_execution
            .map(|device| sample.timings.backend_execution.saturating_sub(device))
    });
    let facade = samples.iter().map(|sample| {
        sample
            .end_to_end
            .saturating_sub(sample.timings.scheduler_queue)
            .saturating_sub(sample.timings.backend_wait)
            .saturating_sub(sample.timings.backend_execution)
    });
    println!("backend={backend}");
    println!(
        "prompt_tokens={} warmup_steps={} measured_steps={} batch_rows=1 batch_wait_us=0 \
         prefix_fill_sessions={} prefix_fill_cooldown_seconds={} clear_allocator_after_fill={} \
         kv_cache_dtype={}{}",
        config.prompt_tokens,
        config.warmup_steps,
        config.measured_steps,
        config.prefix_fill_sessions,
        config.prefix_fill_cooldown_seconds,
        config.clear_allocator_after_fill,
        config.kv_cache_dtype.unwrap_or_default(),
        config.dense_label(),
    );
    println!(
        "decode_tokens_per_second={:.3} end_to_end_mean_ms={:.3} \
         end_to_end_p50_ms={:.3} end_to_end_p95_ms={:.3}",
        rate(samples.len(), total(&end_to_end)),
        mean_ms(end_to_end.iter().copied()),
        percentile_ms(&end_to_end, 50),
        percentile_ms(&end_to_end, 95),
    );
    println!(
        "scheduler_queue_mean_ms={:.3} backend_wait_mean_ms={:.3} \
         backend_execution_mean_ms={:.3} device_execution_mean_ms={} \
         backend_minus_device_mean_ms={} facade_mean_ms={:.3}",
        mean_ms(queue),
        mean_ms(wait),
        mean_ms(backend_wall),
        optional_mean_ms(device),
        optional_mean_ms(backend_minus_device),
        mean_ms(facade),
    );
}

fn fill_prefix_cache(model: &libmir::Model, prompt: &[u32], sessions: usize) -> libmir::Result<()> {
    for offset in 1..=sessions {
        let mut tokens = prompt.to_vec();
        let length = tokens.len();
        tokens.rotate_left(offset % length);
        let mut session = model.session();
        let _output = session.prefill(&tokens, SamplingLogits::None, &mut |_| {})?;
    }
    Ok(())
}

fn total(values: &[Duration]) -> Duration {
    values.iter().copied().sum()
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    u32::try_from(tokens).map_or(f64::INFINITY, f64::from) / elapsed.as_secs_f64()
}

fn mean_ms(values: impl Iterator<Item = Duration>) -> f64 {
    let (seconds, count) = values.fold((0.0, 0_u32), |(seconds, count), value| {
        (seconds + value.as_secs_f64(), count.saturating_add(1))
    });
    seconds * 1_000.0 / f64::from(count)
}

fn optional_mean_ms(values: impl Iterator<Item = Duration>) -> String {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        "unavailable".into()
    } else {
        format!("{:.3}", mean_ms(values.into_iter()))
    }
}

fn percentile_ms(values: &[Duration], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = sorted.len().saturating_mul(percentile).div_ceil(100).saturating_sub(1);
    sorted[rank.min(sorted.len().saturating_sub(1))].as_secs_f64() * 1_000.0
}

fn required_token(token: Option<u32>) -> libmir::Result<u32> {
    token.ok_or_else(|| RuntimeError::Backend("device sampling returned no token".into()).into())
}

fn request(_model: &libmir::Model) -> Conversation {
    Conversation {
        messages: vec![Message {
            role: "user".into(),
            content: "Explain continuous batching in an LLM inference server.".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: libmir::ToolChoice::default(),
    }
}
