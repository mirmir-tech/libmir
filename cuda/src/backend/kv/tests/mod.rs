mod batch;
mod fmha;
mod prefill_batch;
mod split;

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use runtime::kv::{
    BlockId, BlockTable, CacheConfig, KvBackendStorage, KvCacheDType, KvStorageSpec, KvWritePlan,
};
use uuid::Uuid;

use super::super::CudaBackend;
use crate::{CudaConfig, Result};

#[test]
fn paged_gqa_matches_cpu_across_physical_blocks()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let spec = KvStorageSpec::new(
        CacheConfig {
            block_size: 2,
            block_count: 3,
            dtype: KvCacheDType::BFloat16,
        },
        1,
        4,
    );
    let mut cache = backend.prepare_paged_kv(0, spec)?;
    let keys =
        copy(&backend, &bf16s(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))?;
    let values = copy(
        &backend,
        &bf16s(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0]),
    )?;
    let table = block_table();
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, 3)?;
    assert_eq!(cache.store(&plan, &keys, &values)?, 3);

    let query = copy(&backend, &bf16s(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))?;
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 8)?;
    let mut attention = backend.prepare_paged_attention_bf16(&cache, 2, 2)?;
    attention.execute(&query, &cache, &table, &mut output, None, 0.5)?;
    let actual = read(&backend, &output)?;
    let expected = reference(&[0.5, 0.0, 0.5], None);
    assert_output(&actual, &expected);

    attention.execute(&query, &cache, &table, &mut output, Some(2), 0.5)?;
    let actual = read(&backend, &output)?;
    let expected = reference(&[0.0, 0.5], Some(1));
    assert_output(&actual, &expected);
    Ok(())
}

#[test]
fn paged_prefill_is_causal_within_the_query_chunk()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let spec = KvStorageSpec::new(
        CacheConfig {
            block_size: 2,
            block_count: 3,
            dtype: KvCacheDType::BFloat16,
        },
        1,
        4,
    );
    let mut cache = backend.prepare_paged_kv(0, spec)?;
    let keys =
        copy(&backend, &bf16s(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))?;
    let values = copy(
        &backend,
        &bf16s(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0]),
    )?;
    let table = block_table();
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, 3)?;
    cache.store(&plan, &keys, &values)?;
    let query = copy(
        &backend,
        &bf16s(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )?;
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 16)?;
    let mut attention = backend.prepare_paged_attention_bf16(&cache, 2, 2)?;
    attention.execute_prefill(&query, &cache, &table, &mut output, 2, 1, None, 0.5)?;
    let actual = read(&backend, &output)?;
    assert_token_output(&actual, 0, &reference(&[0.5, 0.0], None));
    assert_token_output(&actual, 1, &reference(&[0.5, 0.0, 0.5], None));
    Ok(())
}

#[test]
fn paged_prefill_opens_the_bidirectional_image_block()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let spec = KvStorageSpec::new(
        CacheConfig {
            block_size: 2,
            block_count: 3,
            dtype: KvCacheDType::BFloat16,
        },
        1,
        4,
    );
    let mut cache = backend.prepare_paged_kv(0, spec)?;
    let keys =
        copy(&backend, &bf16s(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))?;
    let values = copy(
        &backend,
        &bf16s(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0]),
    )?;
    let table = block_table();
    cache.store(&KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, 3)?, &keys, &values)?;
    let query_values = [1.0, 0.0, 0.0, 0.0].repeat(6);
    let query = copy(&backend, &bf16s(&query_values))?;
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 24)?;
    backend.prepare_paged_attention_bf16(&cache, 2, 2)?.execute_prefill_masked(
        &query,
        &cache,
        &table,
        &mut output,
        3,
        0,
        None,
        0.5,
        Some((1, 3)),
    )?;
    let actual = read(&backend, &output)?;
    assert_token_output(&actual, 0, &reference(&[0.5], None));
    assert_token_output(&actual, 1, &reference(&[0.5, 0.0, 0.5], None));
    assert_token_output(&actual, 2, &reference(&[0.5, 0.0, 0.5], None));
    Ok(())
}

#[test]
fn paged_fp8_e4m3_quantizes_and_attends_on_device()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let spec = KvStorageSpec::new(
        CacheConfig {
            block_size: 2,
            block_count: 3,
            dtype: KvCacheDType::Fp8E4M3,
        },
        1,
        4,
    );
    let mut cache = backend.prepare_paged_kv(0, spec)?;
    let keys =
        copy(&backend, &bf16s(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))?;
    let values = copy(
        &backend,
        &bf16s(&[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 4.0, 8.0, 12.0, 16.0]),
    )?;
    let table = block_table();
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, 3)?;
    cache.store(&plan, &keys, &values)?;
    let query = copy(&backend, &bf16s(&[1.0, 0.0, 0.0, 0.0]))?;
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 4)?;
    backend
        .prepare_paged_attention_bf16(&cache, 1, 2)?
        .execute(&query, &cache, &table, &mut output, None, 0.5)?;
    let actual = read(&backend, &output)?;
    let expected = weighted_values(&[0.5, 0.0, 0.5]);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual.to_f32() - expected).abs() < 0.25);
    }
    Ok(())
}

fn block_table() -> BlockTable {
    let mut table = BlockTable::with_block_size(2);
    table.push(BlockId(2));
    table.push(BlockId(0));
    table.set_token_len(3);
    table
}

fn reference(scores: &[f32], first_token: Option<usize>) -> [f32; 4] {
    let values = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]];
    let first = first_token.unwrap_or(0);
    let weights: Vec<f32> = scores.iter().map(|score| score.exp()).collect();
    let denominator: f32 = weights.iter().sum();
    let mut output = [0.0; 4];
    for (index, weight) in weights.iter().enumerate() {
        for (target, value) in output.iter_mut().zip(values[first + index]) {
            *target += value * weight / denominator;
        }
    }
    output
}

fn weighted_values(scores: &[f32]) -> [f32; 4] {
    let values = [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [4.0, 8.0, 12.0, 16.0]];
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores.iter().map(|score| (*score - maximum).exp()).collect();
    let denominator: f32 = weights.iter().sum();
    let mut output = [0.0; 4];
    for (weight, values) in weights.iter().zip(values) {
        for (output, value) in output.iter_mut().zip(values) {
            *output += weight * value / denominator;
        }
    }
    output
}

fn assert_output(actual: &[bf16], expected: &[f32; 4]) {
    assert_token_output(actual, 0, expected);
}

fn assert_token_output(actual: &[bf16], token: usize, expected: &[f32; 4]) {
    for head in 0..2 {
        for (column, value) in expected.iter().enumerate() {
            let index = (token * 2 + head) * 4 + column;
            assert!((actual[index].to_f32() - value).abs() < 2.0);
        }
    }
}

fn bf16s(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.synchronize()?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
