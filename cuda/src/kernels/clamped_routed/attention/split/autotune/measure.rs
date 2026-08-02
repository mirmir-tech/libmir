use std::time::Duration;

use mircuda::{Context, DeviceBuffer, Stream, bf16};

use super::super::ClampedRoutedSplitDecode;
use crate::{
    Result,
    kernels::{ClampedRoutedAttention, SplitAttentionWorkspace, SplitPagedAttention},
};

#[allow(clippy::too_many_arguments)]
pub(super) fn measure_direct(
    split: &ClampedRoutedSplitDecode,
    direct: &ClampedRoutedAttention,
    stream: &Stream,
    query: &DeviceBuffer<bf16>,
    key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>,
    table: &DeviceBuffer<u32>,
    sinks: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>,
    blocks: usize,
    window: Option<usize>,
    scale: f32,
    contexts: &[usize],
) -> Result<(Vec<Duration>, Duration)> {
    let mut timings = Vec::with_capacity(contexts.len());
    let mut consumed = Duration::ZERO;
    for &tokens in contexts {
        let (warmup, iterations) = split.backend.auto_tuner().iterations(tokens);
        for _ in 0..warmup {
            direct.execute(
                stream, query, query, query, key_pages, value_pages, table, sinks, output, tokens,
                blocks, window, scale,
            )?;
        }
        let average = measure(split.backend.context(), stream, iterations, || {
            direct.execute(
                stream, query, query, query, key_pages, value_pages, table, sinks, output, tokens,
                blocks, window, scale,
            )
        })?;
        consumed =
            consumed.saturating_add(average.saturating_mul(iterations.saturating_add(warmup)));
        timings.push(average);
    }
    Ok((timings, consumed))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn measure_split(
    split: &ClampedRoutedSplitDecode,
    partition_tokens: usize,
    stream: &Stream,
    query: &DeviceBuffer<bf16>,
    key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>,
    table: &DeviceBuffer<u32>,
    sinks: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>,
    blocks: usize,
    window: Option<usize>,
    scale: f32,
    contexts: &[usize],
) -> Result<(Vec<Duration>, Duration)> {
    let operation = SplitPagedAttention::compile(
        split.backend.compiler(),
        split.operation.spec(),
        partition_tokens,
    )?;
    let (values, statistics) = operation.workspace_lengths();
    let mut workspace = SplitAttentionWorkspace::new(
        split.backend.pool().allocate(stream, values)?,
        split.backend.pool().allocate(stream, statistics)?,
        split.backend.pool().allocate(stream, statistics)?,
    );
    let mut timings = Vec::with_capacity(contexts.len());
    let mut consumed = Duration::ZERO;
    for &tokens in contexts {
        let (warmup, iterations) = split.backend.auto_tuner().iterations(tokens);
        for _ in 0..warmup {
            execute_split(
                split, &operation, &mut workspace, stream, query, key_pages, value_pages, table,
                sinks, output, tokens, blocks, window, scale,
            )?;
        }
        let average = measure(split.backend.context(), stream, iterations, || {
            execute_split(
                split, &operation, &mut workspace, stream, query, key_pages, value_pages, table,
                sinks, output, tokens, blocks, window, scale,
            )
        })?;
        consumed =
            consumed.saturating_add(average.saturating_mul(iterations.saturating_add(warmup)));
        timings.push(average);
    }
    Ok((timings, consumed))
}

#[allow(clippy::too_many_arguments)]
fn execute_split(
    split: &ClampedRoutedSplitDecode,
    operation: &SplitPagedAttention,
    workspace: &mut SplitAttentionWorkspace,
    stream: &Stream,
    query: &DeviceBuffer<bf16>,
    key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>,
    table: &DeviceBuffer<u32>,
    sinks: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>,
    tokens: usize,
    blocks: usize,
    window: Option<usize>,
    scale: f32,
) -> Result<()> {
    let active = operation.execute_partitions(
        stream, query, key_pages, value_pages, table, workspace, output, tokens, blocks, window,
        scale,
    )?;
    let query_heads = sinks.len();
    let head_dim = output.len() / query_heads;
    split.merge.launch(
        stream,
        operation.configs(active)?.merge,
        (
            &workspace.values,
            &workspace.maxima,
            &workspace.denominators,
            sinks,
            output,
            u32::try_from(query_heads)?,
            u32::try_from(head_dim)?,
            u32::try_from(active)?,
            u32::try_from(operation.max_partitions())?,
        ),
    )?;
    Ok(())
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
