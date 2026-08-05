use std::time::{Duration, Instant};

use mircuda::{DeviceBuffer, bf16};

use super::super::{BatchedPagedAttentionBf16, allocate_workspace};
use crate::{
    AttentionExecution, Result,
    backend::{PagedDecodeBatch, PagedKvCache, kv::attention::autotune},
    kernels::BatchedSplitPagedAttention,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn measure(
    attention: &BatchedPagedAttentionBf16,
    query: &DeviceBuffer<bf16>,
    cache: &PagedKvCache,
    batch: &PagedDecodeBatch,
    output: &mut DeviceBuffer<bf16>,
    window: Option<usize>,
    scale: f32,
) -> Result<(AttentionExecution, Duration, Duration)> {
    let tokens = batch.maximum_tokens();
    let visible = window.map_or(tokens, |limit| tokens.min(limit));
    let started = Instant::now();
    let contexts = autotune::sample_contexts(visible, batch.cache_config().block_size);
    let samples = contexts
        .iter()
        .map(|tokens| batch.tuning_sample(&attention.backend, *tokens))
        .collect::<Result<Vec<_>>>()?;
    let direct = measure_direct(attention, query, cache, &samples, output, window, scale)?;
    let mut splits = Vec::new();
    for partition in autotune::candidate_partitions(attention.split.partition_tokens()) {
        let average =
            measure_split(attention, partition, query, cache, &samples, output, window, scale)?;
        splits.push(autotune::SplitMeasurement {
            partition_tokens: partition,
            score: average.iter().copied().sum(),
            timings: average,
        });
    }
    let execution = autotune::select_execution(
        attention.fallback_execution,
        attention.split.partition_tokens(),
        attention.plan_request.max_context_tokens,
        &contexts,
        &direct,
        &splits,
        attention.backend.inner.tuner.minimum_improvement_bps(),
    );
    let average = autotune::execution_average(execution, &direct, &splits);
    Ok((execution, average, started.elapsed()))
}

#[allow(clippy::too_many_arguments)]
fn measure_direct(
    attention: &BatchedPagedAttentionBf16,
    query: &DeviceBuffer<bf16>,
    cache: &PagedKvCache,
    batches: &[PagedDecodeBatch],
    output: &mut DeviceBuffer<bf16>,
    window: Option<usize>,
    scale: f32,
) -> Result<Vec<Duration>> {
    batches
        .iter()
        .map(|batch| {
            let mut execute = || {
                attention.operation.execute(
                    &attention.stream,
                    query,
                    cache.key_pages(),
                    cache.value_pages(),
                    batch.tables(),
                    batch.token_counts(),
                    batch.block_counts(),
                    output,
                    batch.active(),
                    window,
                    scale,
                    attention.plan_request.max_context_tokens + 1,
                )
            };
            measured(attention, batch.maximum_tokens(), &mut execute)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn measure_split(
    attention: &BatchedPagedAttentionBf16,
    partition: usize,
    query: &DeviceBuffer<bf16>,
    cache: &PagedKvCache,
    batches: &[PagedDecodeBatch],
    output: &mut DeviceBuffer<bf16>,
    window: Option<usize>,
    scale: f32,
) -> Result<Vec<Duration>> {
    let split = BatchedSplitPagedAttention::compile(
        &attention.backend.inner.compiler,
        attention.spec(),
        attention.max_batch,
        partition,
    )?;
    let mut workspace = allocate_workspace(&attention.backend, &split)?;
    batches
        .iter()
        .map(|batch| {
            let mut execute = || {
                split.execute(
                    &attention.stream,
                    query,
                    cache.key_pages(),
                    cache.value_pages(),
                    batch.tables(),
                    batch.token_counts(),
                    batch.block_counts(),
                    &mut workspace,
                    output,
                    batch.active(),
                    window,
                    scale,
                    0,
                    batch.maximum_tokens(),
                )
            };
            measured(attention, batch.maximum_tokens(), &mut execute)
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn measured(
    attention: &BatchedPagedAttentionBf16,
    tokens: usize,
    execute: &mut impl FnMut() -> Result<()>,
) -> Result<Duration> {
    let (warmup, iterations) = attention.backend.inner.tuner.iterations(tokens);
    for _ in 0..warmup {
        execute()?;
    }
    let started = attention.backend.inner.context.create_event(true)?;
    let completed = attention.backend.inner.context.create_event(true)?;
    started.record(&attention.stream)?;
    for _ in 0..iterations {
        execute()?;
    }
    completed.record(&attention.stream)?;
    completed.synchronize()?;
    Ok(Duration::from_secs_f32(
        started.elapsed_ms(&completed)? / (iterations as f32 * 1_000.0),
    ))
}
