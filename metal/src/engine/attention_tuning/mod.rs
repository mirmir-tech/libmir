use std::time::{Duration, Instant};

pub(super) use profile::AttentionKey;
use runtime::tuning::{TuningMode, select_fastest_candidate};

use self::profile::{fallback, key};
use super::{
    Result, Stream,
    attention::PagedAttentionScratch,
    kernels::{Kernels, PagedExecution, two_pass_supported},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TuneAction {
    Execute(PagedExecution),
    Measure,
}

pub(super) fn forward(
    kernels: &Kernels,
    inputs: [&mirtal::Array; 5],
    scratch: &PagedAttentionScratch,
    page_size: usize,
    context_tokens: usize,
    scale: f32,
    stream: &Stream,
) -> Result<mirtal::Array> {
    let key = key(inputs, page_size, context_tokens)?;
    let fallback = fallback(key, context_tokens);
    if !two_pass_supported(context_tokens, key.head_dim, key.query_heads, key.kv_heads) {
        return execute(
            kernels,
            PagedExecution::Direct,
            inputs,
            scratch,
            page_size,
            context_tokens,
            scale,
            stream,
        );
    }
    let action = {
        let tuner = stream.tuner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if tuner.config().mode == TuningMode::Disabled {
            TuneAction::Execute(fallback)
        } else if let Some(execution) = tuner.attention_decision(key) {
            TuneAction::Execute(execution)
        } else if tuner.config().mode == TuningMode::Startup && tuner.attention_budget_available() {
            TuneAction::Measure
        } else {
            TuneAction::Execute(fallback)
        }
    };
    match action {
        TuneAction::Execute(execution) => {
            execute(kernels, execution, inputs, scratch, page_size, context_tokens, scale, stream)
        },
        TuneAction::Measure => {
            let started = Instant::now();
            tune(
                kernels, key, fallback, inputs, scratch, page_size, context_tokens, scale, stream,
            )
            .or_else(|error| {
                stream
                    .tuner
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .record_attention(key, fallback, started.elapsed());
                tracing::warn!(
                    target: "libmir::metal::tuning",
                    %error,
                    "Metal paged-attention tuning failed; retaining heuristic execution"
                );
                execute(
                    kernels, fallback, inputs, scratch, page_size, context_tokens, scale, stream,
                )
            })
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn tune(
    kernels: &Kernels,
    key: AttentionKey,
    fallback: PagedExecution,
    inputs: [&mirtal::Array; 5],
    scratch: &PagedAttentionScratch,
    page_size: usize,
    context_tokens: usize,
    scale: f32,
    stream: &Stream,
) -> Result<mirtal::Array> {
    let config = stream.config().tuning.clone();
    let started = Instant::now();
    let candidates = candidates(fallback);
    let timings = candidates
        .iter()
        .copied()
        .map(|execution| {
            measure(kernels, execution, inputs, scratch, page_size, context_tokens, scale, stream)
        })
        .collect::<Result<Vec<_>>>()?;
    let fastest = timings
        .iter()
        .enumerate()
        .min_by_key(|(_, duration)| *duration)
        .map_or(0, |(index, _)| index);
    let fallback_index =
        candidates.iter().position(|candidate| *candidate == fallback).unwrap_or(0);
    let selected =
        select_fastest_candidate(fastest, fallback_index, &timings, config.minimum_improvement_bps);
    let execution = candidates[selected];
    {
        let mut tuner = stream.tuner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        tuner.record_attention(key, execution, started.elapsed());
        tuner.persist();
    }
    tracing::info!(
        target: "libmir::metal::tuning",
        ?execution,
        context_bucket = key.context_bucket,
        context_tokens,
        query_heads = key.query_heads,
        kv_heads = key.kv_heads,
        head_dim = key.head_dim,
        ?candidates,
        timings_us = ?timings
            .iter()
            .map(|duration| duration.as_secs_f64() * 1_000_000.0)
            .collect::<Vec<_>>(),
        "selected Metal paged-attention execution profile"
    );
    execute(kernels, execution, inputs, scratch, page_size, context_tokens, scale, stream)
}

fn candidates(fallback: PagedExecution) -> Vec<PagedExecution> {
    let mut candidates = vec![PagedExecution::Direct];
    let Some(blocks) = fallback.blocks() else {
        return candidates;
    };
    for blocks in [blocks.saturating_div(2).max(32), blocks, blocks.saturating_mul(2).min(1_024)] {
        for reduction_groups in [8, 16, 32] {
            let execution = PagedExecution::TwoPass { blocks, reduction_groups };
            if !candidates.contains(&execution) {
                candidates.push(execution);
            }
        }
    }
    candidates
}

#[allow(clippy::too_many_arguments)]
fn measure(
    kernels: &Kernels,
    execution: PagedExecution,
    inputs: [&mirtal::Array; 5],
    scratch: &PagedAttentionScratch,
    page_size: usize,
    context_tokens: usize,
    scale: f32,
    stream: &Stream,
) -> Result<Duration> {
    let config = &stream.config().tuning;
    for _ in 0..config.warmup_iterations {
        execute(kernels, execution, inputs, scratch, page_size, context_tokens, scale, stream)?
            .async_eval()?;
    }
    stream.synchronize()?;
    let iterations = config.measurement_iterations.max(1);
    let started = Instant::now();
    for _ in 0..iterations {
        execute(kernels, execution, inputs, scratch, page_size, context_tokens, scale, stream)?
            .async_eval()?;
    }
    stream.synchronize()?;
    Ok(started.elapsed() / iterations)
}

#[allow(clippy::too_many_arguments)]
fn execute(
    kernels: &Kernels,
    execution: PagedExecution,
    inputs: [&mirtal::Array; 5],
    scratch: &PagedAttentionScratch,
    page_size: usize,
    context_tokens: usize,
    scale: f32,
    stream: &Stream,
) -> Result<mirtal::Array> {
    kernels.paged_attention(
        stream.native(),
        inputs,
        scratch,
        page_size,
        context_tokens,
        scale,
        execution,
    )
}

mod profile;
#[cfg(test)]
mod tests;
