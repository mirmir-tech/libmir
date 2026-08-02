use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::kv::{
    BlockId, BlockTable, CacheConfig, KvBackendStorage, KvCacheDType, KvStorageSpec, KvWritePlan,
};
use uuid::Uuid;

use super::super::super::CudaBackend;
use crate::{CudaAttentionPolicy, CudaConfig, CudaPlanningPolicy, Result};

const TOKENS: usize = 513;
const BLOCK_SIZE: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;

#[test]
fn split_kv_matches_direct_attention() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig {
        planning: CudaPlanningPolicy {
            attention: CudaAttentionPolicy::SplitKv {
                partition_tokens: 64,
                threshold_tokens: 128,
            },
            ..CudaPlanningPolicy::default()
        },
        ..CudaConfig::default()
    })?;
    for dtype in [KvCacheDType::BFloat16, KvCacheDType::Fp8E4M3] {
        for query_heads in [2, 4, 6, 16] {
            compare(&backend, dtype, query_heads, None)?;
            compare(&backend, dtype, query_heads, Some(300))?;
        }
    }
    Ok(())
}

fn compare(
    backend: &CudaBackend,
    dtype: KvCacheDType,
    query_heads: usize,
    window: Option<usize>,
) -> Result<()> {
    let blocks = TOKENS.div_ceil(BLOCK_SIZE);
    let storage = KvStorageSpec::new(
        CacheConfig {
            block_size: BLOCK_SIZE,
            block_count: u32::try_from(blocks)?,
            dtype,
        },
        KV_HEADS,
        HEAD_DIM,
    );
    let mut cache = backend.prepare_paged_kv(0, storage)?;
    let width = TOKENS * KV_HEADS * HEAD_DIM;
    let keys = copy(backend, &values(width, 29, 14, 31.0)?)?;
    let value_input = copy(backend, &values(width, 37, 18, 23.0)?)?;
    let query = copy(backend, &values(query_heads * HEAD_DIM, 17, 8, 11.0)?)?;
    let table = block_table(blocks)?;
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, TOKENS)?;
    cache.store(&plan, &keys, &value_input)?;
    let output_len = query_heads * HEAD_DIM;
    let mut direct = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, output_len)?;
    let mut split = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, output_len)?;
    let mut attention = backend.prepare_paged_attention_bf16(&cache, query_heads, blocks)?;
    attention.execute_direct(&query, &cache, &table, &mut direct, window, 0.25)?;
    attention.execute_split(&query, &cache, &table, &mut split, window, 0.25)?;
    let direct = read(backend, &direct)?;
    let split = read(backend, &split)?;
    let max_error = direct
        .iter()
        .zip(split)
        .map(|(direct, split)| (direct.to_f32() - split.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_error <= 0.015_625,
        "{dtype:?} GQA {query_heads}/{KV_HEADS} {window:?} max error: {max_error}"
    );
    Ok(())
}

fn values(len: usize, modulus: usize, midpoint: usize, divisor: f32) -> Result<Vec<bf16>> {
    (0..len)
        .map(|index| {
            let value = i16::try_from(index % modulus)? - i16::try_from(midpoint)?;
            Ok(bf16::from_f32(f32::from(value) / divisor))
        })
        .collect()
}

fn block_table(blocks: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in 0..blocks {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(TOKENS);
    Ok(table)
}

fn copy(backend: &CudaBackend, values: &[bf16]) -> Result<DeviceBuffer<bf16>> {
    let mut host = backend.inner.context.allocate_pinned::<bf16>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read(backend: &CudaBackend, source: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host: PinnedBuffer<bf16> = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    backend.inner.stream.synchronize()?;
    Ok(host.to_vec()?)
}
