//! Gemma 2's attention score cap through the direct, split and prefill paged
//! kernels against a host reference that caps the same scores.

use mircuda::bf16;
use runtime::kv::{CacheConfig, KvBackendStorage, KvCacheDType, KvStorageSpec, KvWritePlan};
use uuid::Uuid;

use super::{
    super::super::{CudaBackend, kv::PagedAttentionBf16},
    split::{block_table, copy, read, values},
};
use crate::{CudaAttentionPolicy, CudaConfig, CudaPlanningPolicy, Result, kernels::ScoreSoftcap};

const TOKENS: usize = 513;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const CAP: f32 = 1.5;
const SCALE: f32 = 0.25;
const PREFILL: usize = 4;

#[test]
fn capped_scores_match_a_capped_reference() -> Result<()> {
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
    let blocks = TOKENS.div_ceil(super::split::BLOCK_SIZE);
    let storage = KvStorageSpec::new(
        CacheConfig {
            block_size: super::split::BLOCK_SIZE,
            block_count: u32::try_from(blocks)?,
            dtype: KvCacheDType::BFloat16,
        },
        KV_HEADS,
        HEAD_DIM,
    );
    let mut cache = backend.prepare_paged_kv(0, storage)?;
    let width = TOKENS * KV_HEADS * HEAD_DIM;
    let (keys, value_rows) = (values(width, 29, 14, 7.0)?, values(width, 37, 18, 23.0)?);
    let table = block_table(blocks)?;
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, TOKENS)?;
    cache.store(&plan, &copy(&backend, &keys)?, &copy(&backend, &value_rows)?)?;
    let cap = Some(ScoreSoftcap::new(CAP)?);
    for query_heads in [2, 8] {
        let mut attention = PagedAttentionBf16::new(&backend, &cache, query_heads, blocks, cap)?;
        let queries = values(PREFILL * query_heads * HEAD_DIM, 17, 8, 3.0)?;
        let last = &queries[(PREFILL - 1) * query_heads * HEAD_DIM..];
        let expected = reference(last, &keys, &value_rows, query_heads, TOKENS);
        let query = copy(&backend, last)?;
        let length = query_heads * HEAD_DIM;
        let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, length)?;
        attention.execute_direct(&query, &cache, &table, &mut output, None, SCALE)?;
        compare(&expected, &read(&backend, &output)?, &format!("direct {query_heads}"));
        attention.execute_split(&query, &cache, &table, &mut output, None, SCALE)?;
        compare(&expected, &read(&backend, &output)?, &format!("split {query_heads}"));
        let query = copy(&backend, &queries)?;
        let mut output =
            backend.inner.pool.allocate::<bf16>(&backend.inner.stream, PREFILL * length)?;
        let start = TOKENS - PREFILL;
        attention
            .execute_prefill(&query, &cache, &table, &mut output, PREFILL, start, None, SCALE)?;
        let actual = read(&backend, &output)?;
        for row in 0..PREFILL {
            let query = &queries[row * length..(row + 1) * length];
            let expected = reference(query, &keys, &value_rows, query_heads, start + row + 1);
            let label = format!("prefill {query_heads} row {row}");
            compare(&expected, &actual[row * length..(row + 1) * length], &label);
        }
    }
    Ok(())
}

/// Every head's capped attention over the first `context` tokens.
fn reference(
    query: &[bf16],
    keys: &[bf16],
    values: &[bf16],
    query_heads: usize,
    context: usize,
) -> Vec<f32> {
    let group = query_heads / KV_HEADS;
    let mut output = Vec::with_capacity(query_heads * HEAD_DIM);
    for head in 0..query_heads {
        let kv = head / group;
        let row = |data: &[bf16], token: usize| {
            let start = (token * KV_HEADS + kv) * HEAD_DIM;
            data[start..start + HEAD_DIM]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>()
        };
        let q = &query[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let scores = (0..context)
            .map(|token| {
                let dot: f32 = q.iter().zip(row(keys, token)).map(|(q, k)| q.to_f32() * k).sum();
                CAP * (dot * SCALE / CAP).tanh()
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = scores.iter().map(|score| (score - maximum).exp()).collect::<Vec<_>>();
        let total: f32 = weights.iter().sum();
        for column in 0..HEAD_DIM {
            let value: f32 =
                (0..context).map(|token| weights[token] * row(values, token)[column]).sum();
            output.push(value / total);
        }
    }
    output
}

fn compare(expected: &[f32], actual: &[bf16], label: &str) {
    let error = expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| (expected - actual.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(error <= 0.02, "{label}: max error {error}");
}
