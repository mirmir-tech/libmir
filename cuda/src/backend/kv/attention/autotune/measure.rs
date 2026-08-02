use std::time::Duration;

use ::runtime::kv::BlockTable;
use mircuda::{Context, DeviceBuffer, Stream, bf16};

use super::super::{PagedAttentionBf16, PagedKvCache};
use crate::{
    Result,
    kernels::{SplitAttentionWorkspace, SplitPagedAttention},
};

#[allow(clippy::too_many_arguments)]
pub(super) fn measure_direct(
    attention: &PagedAttentionBf16,
    query: &DeviceBuffer<bf16>,
    cache: &PagedKvCache,
    table: &BlockTable,
    output: &mut DeviceBuffer<bf16>,
    window: Option<usize>,
    scale: f32,
    contexts: &[usize],
) -> Result<(Vec<Duration>, Duration)> {
    let mut timings = Vec::with_capacity(contexts.len());
    let mut consumed = Duration::ZERO;
    for &tokens in contexts {
        let (warmup, iterations) = attention.tuner.iterations(tokens);
        for _ in 0..warmup {
            execute_direct(attention, query, cache, table, output, tokens, window, scale)?;
        }
        let average = measure(&attention.context, &attention.stream, iterations, || {
            execute_direct(attention, query, cache, table, output, tokens, window, scale)
        })?;
        consumed =
            consumed.saturating_add(average.saturating_mul(iterations.saturating_add(warmup)));
        timings.push(average);
    }
    Ok((timings, consumed))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn measure_split(
    attention: &PagedAttentionBf16,
    partition_tokens: usize,
    query: &DeviceBuffer<bf16>,
    cache: &PagedKvCache,
    table: &BlockTable,
    output: &mut DeviceBuffer<bf16>,
    window: Option<usize>,
    scale: f32,
    contexts: &[usize],
) -> Result<(Vec<Duration>, Duration)> {
    let split = SplitPagedAttention::compile(
        &attention.backend.inner.compiler,
        attention.spec,
        partition_tokens,
    )?;
    let (values, statistics) = split.workspace_lengths();
    let mut workspace = SplitAttentionWorkspace::new(
        attention.pool.allocate(&attention.stream, values)?,
        attention.pool.allocate(&attention.stream, statistics)?,
        attention.pool.allocate(&attention.stream, statistics)?,
    );
    let mut timings = Vec::with_capacity(contexts.len());
    let mut consumed = Duration::ZERO;
    for &tokens in contexts {
        let (warmup, iterations) = attention.tuner.iterations(tokens);
        for _ in 0..warmup {
            execute_split(
                attention, &split, &mut workspace, query, cache, table, output, tokens, window,
                scale,
            )?;
        }
        let average = measure(&attention.context, &attention.stream, iterations, || {
            execute_split(
                attention, &split, &mut workspace, query, cache, table, output, tokens, window,
                scale,
            )
        })?;
        consumed =
            consumed.saturating_add(average.saturating_mul(iterations.saturating_add(warmup)));
        timings.push(average);
    }
    Ok((timings, consumed))
}

#[allow(clippy::cast_precision_loss)]
fn measure(
    context: &Context,
    stream: &Stream,
    iterations: u32,
    mut execute: impl FnMut() -> Result<()>,
) -> Result<Duration> {
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..iterations {
        execute()?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(Duration::from_secs_f32(
        started.elapsed_ms(&completed)? / (iterations as f32 * 1_000.0),
    ))
}

#[allow(clippy::too_many_arguments)]
fn execute_direct(
    attention: &PagedAttentionBf16,
    query: &DeviceBuffer<bf16>,
    cache: &PagedKvCache,
    table: &BlockTable,
    output: &mut DeviceBuffer<bf16>,
    tokens: usize,
    window: Option<usize>,
    scale: f32,
) -> Result<()> {
    attention.operation.execute(
        &attention.stream,
        query,
        cache.key_pages(),
        cache.value_pages(),
        &attention.table_device,
        output,
        tokens,
        table.blocks().len(),
        window,
        scale,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_split(
    attention: &PagedAttentionBf16,
    split: &SplitPagedAttention,
    workspace: &mut SplitAttentionWorkspace,
    query: &DeviceBuffer<bf16>,
    cache: &PagedKvCache,
    table: &BlockTable,
    output: &mut DeviceBuffer<bf16>,
    tokens: usize,
    window: Option<usize>,
    scale: f32,
) -> Result<()> {
    split.execute(
        &attention.stream,
        query,
        cache.key_pages(),
        cache.value_pages(),
        &attention.table_device,
        workspace,
        output,
        tokens,
        table.blocks().len(),
        window,
        scale,
    )
}

pub fn sample_contexts(visible: usize, block_size: usize) -> Vec<usize> {
    let first = visible.min(block_size.max(64));
    let mut contexts = vec![first, visible.div_ceil(4), visible.div_ceil(2), visible];
    contexts.retain(|tokens| *tokens > 0);
    contexts.sort_unstable();
    contexts.dedup();
    contexts
}

pub fn candidate_partitions(fallback: usize) -> Vec<usize> {
    let mut candidates = vec![64, 128, 256, 384, 512, fallback];
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

#[cfg(test)]
mod tests {
    use super::sample_contexts;

    #[test]
    fn samples_short_and_long_observed_contexts() {
        assert_eq!(sample_contexts(1_024, 16), vec![64, 256, 512, 1_024]);
        assert_eq!(sample_contexts(17, 16), vec![5, 9, 17]);
    }
}
