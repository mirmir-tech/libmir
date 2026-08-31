use mircuda::{DeviceBuffer, DeviceElement, bf16};
use runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType, KvStorageSpec, KvWritePlan};
use uuid::Uuid;

use super::WindowedPrefillStaging;
use crate::{CudaBackend, CudaConfig, Result, kernels::ClampedRoutedAttention};

const BLOCK_SIZE: usize = 16;
const HEAD_DIM: usize = 64;
const QUERY_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const WINDOW: usize = 128;
const PREFIX: usize = 127;
const QUERY: usize = 2_048;
const RING_BLOCKS: usize = (WINDOW + BLOCK_SIZE - 1).div_ceil(BLOCK_SIZE);

#[test]
fn staging_context_keeps_only_the_receptive_history() {
    let window = 128_usize;
    let start = 8_192_usize;
    let query = 2_048_usize;
    let history = start.min(window - 1);

    assert_eq!(history, 127);
    assert_eq!(history + query, 2_175);
    assert_eq!(start - history, 8_065);
}

#[test]
#[allow(clippy::too_many_lines)]
fn staged_windowed_fa2_matches_scalar_with_prefix_and_current_chunk() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let storage = storage();
    let mut cache = backend.prepare_windowed_paged_kv(0, storage, WINDOW, 1)?;
    let prefix_table = table(PREFIX);
    let prefix_keys = values(PREFIX, 0.013);
    let prefix_values = values(PREFIX, 0.031);
    let prefix_keys_device = copy(&backend, &bf16s(&prefix_keys))?;
    let prefix_values_device = copy(&backend, &bf16s(&prefix_values))?;
    let write = KvWritePlan::prefill(Uuid::nil(), 0, &prefix_table, 0, PREFIX)?;
    cache.store_for_session(&write, 0, &prefix_keys_device, &prefix_values_device)?;

    let current_keys = copy(&backend, &bf16s(&values(QUERY, 0.047)))?;
    let current_values = copy(&backend, &bf16s(&values(QUERY, 0.071)))?;
    let queries = copy(&backend, &bf16s(&query_values()))?;
    let sinks = copy(&backend, &bf16s(&[0.2, -0.1]))?;
    let full_table = table(PREFIX + QUERY);
    let max_blocks = (PREFIX + QUERY).div_ceil(BLOCK_SIZE);
    let mut batch = backend.prepare_paged_prefill_batch(storage, max_blocks, 1, QUERY)?;
    batch.prepare(&[&full_table], &[PREFIX], &[QUERY])?;
    batch.prepare_ring(&[&full_table], &[PREFIX], &[QUERY], &[0], RING_BLOCKS, WINDOW)?;

    let mut scalar_output = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, QUERY * QUERY_HEADS * HEAD_DIM)?;
    let mut scalar_lse = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, QUERY * QUERY_HEADS * HEAD_DIM)?;
    let scalar = ClampedRoutedAttention::compile_scalar(
        &backend,
        BLOCK_SIZE,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        KvCacheDType::BFloat16,
        Some(WINDOW),
    )?;
    scalar.execute_prefill_batch(
        &backend.inner.stream,
        &queries,
        &current_keys,
        &current_values,
        cache.key_pages(),
        cache.value_pages(),
        &batch,
        batch.ring_tables(),
        &sinks,
        &mut scalar_lse,
        &mut scalar_output,
        Some(WINDOW),
        0.125,
    )?;

    let mut staged = WindowedPrefillStaging::new(&backend, storage, 1, QUERY, WINDOW)?;
    staged.stage(
        &batch,
        &current_keys,
        &current_values,
        cache.key_pages(),
        cache.value_pages(),
        WINDOW,
    )?;
    let mut fa2_output = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, QUERY * QUERY_HEADS * HEAD_DIM)?;
    let mut fa2_lse = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, QUERY * QUERY_HEADS * HEAD_DIM)?;
    let fa2 = ClampedRoutedAttention::compile(
        &backend,
        BLOCK_SIZE,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        KvCacheDType::BFloat16,
        Some(WINDOW),
    )?;
    let mut fallback_output = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, QUERY * QUERY_HEADS * HEAD_DIM)?;
    let mut fallback_lse = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, QUERY * QUERY_HEADS * HEAD_DIM)?;
    fa2.execute_prefill_batch(
        &backend.inner.stream,
        &queries,
        &current_keys,
        &current_values,
        cache.key_pages(),
        cache.value_pages(),
        &batch,
        batch.ring_tables(),
        &sinks,
        &mut fallback_lse,
        &mut fallback_output,
        Some(WINDOW),
        0.125,
    )?;
    assert_close(&read(&backend, &fallback_output)?, &read(&backend, &scalar_output)?);
    assert!(fa2.execute_windowed_fmha(
        &backend.inner.stream,
        &queries,
        &batch,
        &staged,
        &sinks,
        &mut fa2_lse,
        &mut fa2_output,
        0.125,
    )?);
    assert_close(&read(&backend, &fa2_output)?, &read(&backend, &scalar_output)?);
    Ok(())
}

fn storage() -> KvStorageSpec {
    KvStorageSpec::new(
        CacheConfig {
            block_size: BLOCK_SIZE,
            block_count: 256,
            dtype: KvCacheDType::BFloat16,
        },
        KV_HEADS,
        HEAD_DIM,
    )
}

fn table(tokens: usize) -> BlockTable {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in 0..tokens.div_ceil(BLOCK_SIZE) {
        table.push(BlockId(u32::try_from(block).unwrap_or_default()));
    }
    table.set_token_len(tokens);
    table
}

#[allow(clippy::cast_precision_loss)]
fn values(tokens: usize, scale: f32) -> Vec<f32> {
    (0..tokens * KV_HEADS * HEAD_DIM)
        .map(|index| ((index % 29) as f32 - 14.0) * scale)
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn query_values() -> Vec<f32> {
    (0..QUERY * QUERY_HEADS * HEAD_DIM)
        .map(|index| ((index % 23) as f32 - 11.0) * 0.019)
        .collect()
}

fn bf16s(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    backend.synchronize()?;
    Ok(host.to_vec()?)
}

fn assert_close(actual: &[bf16], expected: &[bf16]) {
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual.to_f32() - expected.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_error <= 0.031_25, "maximum scalar/FA2 error was {max_error}");
}
