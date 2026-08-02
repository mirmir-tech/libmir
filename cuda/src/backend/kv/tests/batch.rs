use mircuda::{DeviceBuffer, DeviceElement, bf16};
use runtime::kv::{
    BlockId, BlockTable, CacheConfig, KvBackendStorage, KvCacheDType, KvStorageSpec, KvWritePlan,
};
use uuid::Uuid;

use super::super::super::CudaBackend;
use crate::{CudaConfig, Result};

const BLOCK_SIZE: usize = 2;
const HEAD_DIM: usize = 4;
const QUERY_HEADS: usize = 2;

#[test]
fn batched_paged_attention_matches_independent_sequences() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let storage = KvStorageSpec::new(
        CacheConfig {
            block_size: BLOCK_SIZE,
            block_count: 4,
            dtype: KvCacheDType::BFloat16,
        },
        1,
        HEAD_DIM,
    );
    let mut cache = backend.prepare_paged_kv(0, storage)?;
    let first = table(&[2, 0], 3);
    let second = table(&[3], 2);
    store(
        &backend,
        &mut cache,
        &first,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0],
    )?;
    store(
        &backend,
        &mut cache,
        &second,
        &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        &[5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0],
    )?;

    let first_query = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let second_query = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let query = copy(&backend, &bf16s(&[first_query, second_query].concat()))?;
    let first_device = copy(&backend, &bf16s(&first_query))?;
    let second_device = copy(&backend, &bf16s(&second_query))?;
    let mut expected_first = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    let mut expected_second = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    let mut scalar = backend.prepare_paged_attention_bf16(&cache, QUERY_HEADS, 2)?;
    scalar.execute(&first_device, &cache, &first, &mut expected_first, None, 0.5)?;
    scalar.execute(&second_device, &cache, &second, &mut expected_second, None, 0.5)?;

    let mut actual = backend.inner.pool.allocate(&backend.inner.stream, 16)?;
    let mut batch = backend.prepare_paged_decode_batch(storage, 2, 4)?;
    batch.prepare(&[&first, &second])?;
    backend
        .prepare_batched_paged_attention_bf16(&cache, QUERY_HEADS, 2, 4)?
        .execute(&query, &cache, &batch, &mut actual, None, 0.5)?;
    let expected = [read(&backend, &expected_first)?, read(&backend, &expected_second)?].concat();
    let actual = read(&backend, &actual)?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual.to_f32() - expected.to_f32()).abs() <= 0.015_625);
    }
    Ok(())
}

#[test]
fn batched_split_attention_matches_independent_long_sequences() -> Result<()> {
    const TOKENS: usize = 513;
    const LONG_BLOCK_SIZE: usize = 16;
    const LONG_HEAD_DIM: usize = 16;
    const LONG_KV_HEADS: usize = 2;
    const LONG_QUERY_HEADS: usize = 8;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let blocks = TOKENS.div_ceil(LONG_BLOCK_SIZE);
    let storage = KvStorageSpec::new(
        CacheConfig {
            block_size: LONG_BLOCK_SIZE,
            block_count: u32::try_from(blocks * 2)?,
            dtype: KvCacheDType::BFloat16,
        },
        LONG_KV_HEADS,
        LONG_HEAD_DIM,
    );
    let mut cache = backend.prepare_paged_kv(0, storage)?;
    let first = long_table(0, blocks, TOKENS)?;
    let second = long_table(blocks, blocks, TOKENS)?;
    let width = TOKENS * LONG_KV_HEADS * LONG_HEAD_DIM;
    store(
        &backend,
        &mut cache,
        &first,
        &pattern(width, 29, 14, 31.0)?,
        &pattern(width, 37, 18, 23.0)?,
    )?;
    store(
        &backend,
        &mut cache,
        &second,
        &pattern(width, 31, 15, 29.0)?,
        &pattern(width, 41, 20, 19.0)?,
    )?;
    let query_width = LONG_QUERY_HEADS * LONG_HEAD_DIM;
    let first_query = pattern(query_width, 17, 8, 11.0)?;
    let second_query = pattern(query_width, 19, 9, 13.0)?;
    let query = copy(&backend, &bf16s(&[first_query.clone(), second_query.clone()].concat()))?;
    let first_query = copy(&backend, &bf16s(&first_query))?;
    let second_query = copy(&backend, &bf16s(&second_query))?;
    let mut expected_first = backend.inner.pool.allocate(&backend.inner.stream, query_width)?;
    let mut expected_second = backend.inner.pool.allocate(&backend.inner.stream, query_width)?;
    let mut scalar = backend.prepare_paged_attention_bf16(&cache, LONG_QUERY_HEADS, blocks)?;
    scalar.execute_direct(&first_query, &cache, &first, &mut expected_first, None, 0.25)?;
    scalar.execute_direct(&second_query, &cache, &second, &mut expected_second, None, 0.25)?;
    let mut actual = backend.inner.pool.allocate(&backend.inner.stream, query_width * 2)?;
    let mut batch = backend.prepare_paged_decode_batch(storage, blocks, 2)?;
    batch.prepare(&[&first, &second])?;
    backend
        .prepare_batched_paged_attention_bf16(&cache, LONG_QUERY_HEADS, blocks, 2)?
        .execute(&query, &cache, &batch, &mut actual, None, 0.25)?;
    let expected = [read(&backend, &expected_first)?, read(&backend, &expected_second)?].concat();
    let actual = read(&backend, &actual)?;
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual.to_f32() - expected.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(error <= 0.015_625, "batched split maximum BF16 difference: {error}");
    Ok(())
}

fn table(blocks: &[u32], tokens: usize) -> BlockTable {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in blocks {
        table.push(BlockId(*block));
    }
    table.set_token_len(tokens);
    table
}

fn long_table(first: usize, blocks: usize, tokens: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(16);
    for block in first..first + blocks {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(tokens);
    Ok(table)
}

fn pattern(len: usize, modulus: usize, midpoint: usize, divisor: f32) -> Result<Vec<f32>> {
    (0..len)
        .map(|index| {
            let value = i16::try_from(index % modulus)? - i16::try_from(midpoint)?;
            Ok(f32::from(value) / divisor)
        })
        .collect()
}

fn store(
    backend: &CudaBackend,
    cache: &mut super::super::PagedKvCache,
    table: &BlockTable,
    keys: &[f32],
    values: &[f32],
) -> Result<()> {
    let keys = copy(backend, &bf16s(keys))?;
    let values = copy(backend, &bf16s(values))?;
    let plan = KvWritePlan::prefill(Uuid::new_v4(), 0, table, 0, table.token_len())?;
    cache.store(&plan, &keys, &values)?;
    Ok(())
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
