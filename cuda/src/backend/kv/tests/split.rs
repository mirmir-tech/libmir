use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use models::semantic::AttentionWindow;
use runtime::kv::{
    BlockId, BlockTable, CacheConfig, KvBackendStorage, KvCacheDType, KvStorageSpec, KvWritePlan,
};
use uuid::Uuid;

use super::super::super::CudaBackend;
use crate::{CudaAttentionPolicy, CudaConfig, CudaPlanningPolicy, Result};

const TOKENS: usize = 513;
pub(super) const BLOCK_SIZE: usize = 16;
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
            compare(&backend, dtype, query_heads, Some(AttentionWindow::Sliding(300)))?;
            // The newest of 513 tokens starts its 300- and 380-token chunks at
            // 300 and 380, past the split threshold of 128 visible keys.
            for chunk in [300, 380] {
                compare(&backend, dtype, query_heads, Some(AttentionWindow::Chunked { chunk }))?;
            }
        }
    }
    Ok(())
}

fn compare(
    backend: &CudaBackend,
    dtype: KvCacheDType,
    query_heads: usize,
    window: Option<AttentionWindow>,
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
    let (keys_host, values_host, query_host) = (
        values(width, 29, 14, 31.0)?,
        values(width, 37, 18, 23.0)?,
        values(query_heads * HEAD_DIM, 17, 8, 11.0)?,
    );
    let keys = copy(backend, &keys_host)?;
    let value_input = copy(backend, &values_host)?;
    let query = copy(backend, &query_host)?;
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
        .zip(&split)
        .map(|(direct, split)| (direct.to_f32() - split.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    // BF16 pages hold the inputs exactly, so both paths match an `f32` host
    // reference to accumulation order and the tensor-core path's BF16
    // weights: a sixteenth of the output's scale, where reading the wrong
    // keys misses by the whole output.
    if dtype == KvCacheDType::BFloat16 {
        let expected = host_reference(&keys_host, &values_host, &query_host, query_heads, window);
        let tolerance = expected.iter().fold(0.0_f32, |scale, value| scale.max(value.abs())) / 16.0;
        for (path, actual) in [("direct", &direct), ("split", &split)] {
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual.to_f32() - expected).abs())
                .fold(0.0_f32, f32::max);
            assert!(error <= tolerance, "{path} GQA {query_heads} {window:?} error: {error}");
        }
    }
    assert!(
        max_error <= 0.015_625,
        "{dtype:?} GQA {query_heads}/{KV_HEADS} {window:?} max error: {max_error}"
    );
    Ok(())
}

pub(super) fn values(
    len: usize,
    modulus: usize,
    midpoint: usize,
    divisor: f32,
) -> Result<Vec<bf16>> {
    (0..len)
        .map(|index| {
            let value = i16::try_from(index % modulus)? - i16::try_from(midpoint)?;
            Ok(bf16::from_f32(f32::from(value) / divisor))
        })
        .collect()
}

pub(super) fn block_table(blocks: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in 0..blocks {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(TOKENS);
    Ok(table)
}

pub(super) fn copy(backend: &CudaBackend, values: &[bf16]) -> Result<DeviceBuffer<bf16>> {
    let mut host = backend.inner.context.allocate_pinned::<bf16>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

pub(super) fn read(backend: &CudaBackend, source: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host: PinnedBuffer<bf16> = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    backend.inner.stream.synchronize()?;
    Ok(host.to_vec()?)
}

fn host_reference(
    keys: &[bf16],
    values: &[bf16],
    query: &[bf16],
    query_heads: usize,
    window: Option<AttentionWindow>,
) -> Vec<f32> {
    let visible = crate::kernels::attention_window::visible(TOKENS, window);
    let first = TOKENS - visible;
    let mut output = vec![0.0_f32; query_heads * HEAD_DIM];
    for head in 0..query_heads {
        let kv = head / (query_heads / KV_HEADS);
        let scores = (first..TOKENS)
            .map(|token| {
                (0..HEAD_DIM)
                    .map(|d| {
                        query[head * HEAD_DIM + d].to_f32()
                            * keys[(token * KV_HEADS + kv) * HEAD_DIM + d].to_f32()
                    })
                    .sum::<f32>()
                    * 0.25
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = scores.iter().map(|score| (score - maximum).exp()).collect::<Vec<_>>();
        let total = weights.iter().sum::<f32>();
        for (index, weight) in weights.iter().enumerate() {
            let token = first + index;
            for d in 0..HEAD_DIM {
                output[head * HEAD_DIM + d] +=
                    values[(token * KV_HEADS + kv) * HEAD_DIM + d].to_f32() * weight / total;
            }
        }
    }
    output
}
