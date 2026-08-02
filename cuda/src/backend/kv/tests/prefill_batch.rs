use mircuda::{DeviceBuffer, DeviceElement, bf16};
use runtime::kv::{
    BlockId, BlockTable, CacheConfig, KvBackendStorage, KvCacheDType, KvStorageSpec, KvWritePlan,
};
use uuid::Uuid;

use super::super::super::CudaBackend;
use crate::{CudaConfig, Result};

const BLOCK_SIZE: usize = 2;
const HEAD_DIM: usize = 4;

#[test]
fn packed_prefill_store_matches_independent_sequences() -> Result<()> {
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
    let first = table(&[2, 0], 3);
    let second = table(&[3], 2);
    let keys = values(5, 3.0)?;
    let values = values(5, 7.0)?;
    let mut expected = backend.prepare_paged_kv(0, storage)?;
    store(&backend, &mut expected, &first, &keys[..12], &values[..12])?;
    store(&backend, &mut expected, &second, &keys[12..], &values[12..])?;
    let mut actual = backend.prepare_paged_kv(0, storage)?;
    let keys_device = copy(&backend, &bf16s(&keys))?;
    let values_device = copy(&backend, &bf16s(&values))?;
    let mut prefill = backend.prepare_paged_prefill_batch(storage, 2, 2, 8)?;
    prefill.prepare(&[&first, &second], &[0, 0], &[3, 2])?;
    actual.store_prefill_batch(&prefill, &keys_device, &values_device)?;
    compare_prefill_attention(
        &backend,
        &expected,
        &actual,
        &prefill,
        [&first, &second],
        &keys,
        &keys_device,
    )?;

    let queries = copy(&backend, &bf16s(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))?;
    let mut expected_output = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    let mut actual_output = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    let mut decode = backend.prepare_paged_decode_batch(storage, 2, 2)?;
    decode.prepare(&[&first, &second])?;
    backend.prepare_batched_paged_attention_bf16(&expected, 1, 2, 2)?.execute(
        &queries,
        &expected,
        &decode,
        &mut expected_output,
        None,
        0.5,
    )?;
    backend.prepare_batched_paged_attention_bf16(&actual, 1, 2, 2)?.execute(
        &queries,
        &actual,
        &decode,
        &mut actual_output,
        None,
        0.5,
    )?;
    let expected = read(&backend, &expected_output)?;
    let actual = read(&backend, &actual_output)?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual.to_f32() - expected.to_f32()).abs() <= 0.015_625);
    }
    Ok(())
}

fn compare_prefill_attention(
    backend: &CudaBackend,
    expected: &super::super::PagedKvCache,
    actual: &super::super::PagedKvCache,
    batch: &super::super::PagedPrefillBatch,
    tables: [&BlockTable; 2],
    query: &[f32],
    packed_query: &DeviceBuffer<bf16>,
) -> Result<()> {
    let first_query = copy(backend, &bf16s(&query[..12]))?;
    let second_query = copy(backend, &bf16s(&query[12..]))?;
    let mut first_output = backend.inner.pool.allocate(&backend.inner.stream, 12)?;
    let mut second_output = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    let mut scalar = backend.prepare_paged_attention_bf16(expected, 1, 2)?;
    scalar.execute_prefill(
        &first_query,
        expected,
        tables[0],
        &mut first_output,
        3,
        0,
        None,
        0.5,
    )?;
    scalar.execute_prefill(
        &second_query,
        expected,
        tables[1],
        &mut second_output,
        2,
        0,
        None,
        0.5,
    )?;
    let expected = [read(backend, &first_output)?, read(backend, &second_output)?].concat();
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 20)?;
    backend
        .prepare_batched_paged_prefill_attention_bf16(actual, 1, 2, 2)?
        .execute(packed_query, actual, batch, &mut output, None, 0.5)?;
    let actual = read(backend, &output)?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual.to_f32() - expected.to_f32()).abs() <= 0.015_625);
    }
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

fn values(tokens: usize, divisor: f32) -> Result<Vec<f32>> {
    (0..tokens * HEAD_DIM)
        .map(|index| Ok(f32::from(u16::try_from(index)?) / divisor))
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
