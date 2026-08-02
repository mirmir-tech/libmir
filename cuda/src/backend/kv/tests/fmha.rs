use runtime::kv::{
    BlockId, BlockTable, CacheConfig, KvBackendStorage, KvCacheDType, KvStorageSpec, KvWritePlan,
};
use uuid::Uuid;

use super::{bf16s, copy, read};
use crate::{CudaBackend, CudaConfig, Result};

const CONTEXT: usize = 64;
const QUERY_TOKENS: usize = 32;
const QUERY_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;

#[test]
fn contiguous_bf16_prefill_uses_gqa_fmha() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let blocks = CONTEXT.div_ceil(BLOCK_SIZE);
    let storage = KvStorageSpec::new(
        CacheConfig {
            block_size: BLOCK_SIZE,
            block_count: u32::try_from(blocks)?,
            dtype: KvCacheDType::BFloat16,
        },
        KV_HEADS,
        HEAD_DIM,
    );
    let mut cache = backend.prepare_paged_kv(0, storage)?;
    let keys = copy(&backend, &bf16s(&vec![0.0; CONTEXT * KV_HEADS * HEAD_DIM]))?;
    let values = (0..CONTEXT * KV_HEADS * HEAD_DIM)
        .map(|index| [1.0_f32, 2.0_f32][(index / HEAD_DIM) % KV_HEADS])
        .collect::<Vec<_>>();
    let values = copy(&backend, &bf16s(&values))?;
    let table = block_table(blocks)?;
    cache.store(&KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, CONTEXT)?, &keys, &values)?;
    let query = copy(&backend, &bf16s(&vec![0.0; QUERY_TOKENS * QUERY_HEADS * HEAD_DIM]))?;
    let mut output = backend
        .inner
        .pool
        .allocate::<mircuda::bf16>(&backend.inner.stream, QUERY_TOKENS * QUERY_HEADS * HEAD_DIM)?;
    backend
        .prepare_paged_attention_bf16(&cache, QUERY_HEADS, blocks)?
        .execute_prefill(
            &query,
            &cache,
            &table,
            &mut output,
            QUERY_TOKENS,
            CONTEXT - QUERY_TOKENS,
            None,
            1.0 / 128.0_f32.sqrt(),
        )?;
    let output = read(&backend, &output)?;
    for (index, actual) in output.iter().enumerate() {
        let head = (index / HEAD_DIM) % QUERY_HEADS;
        let expected = f32::from(u8::try_from(head / (QUERY_HEADS / KV_HEADS) + 1)?);
        assert!((actual.to_f32() - expected).abs() <= 0.015_625);
    }
    Ok(())
}

#[test]
fn contiguous_fmha_matches_scattered_paged_prefill() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let blocks = CONTEXT.div_ceil(BLOCK_SIZE);
    let storage = KvStorageSpec::new(
        CacheConfig {
            block_size: BLOCK_SIZE,
            block_count: u32::try_from(blocks * 2)?,
            dtype: KvCacheDType::BFloat16,
        },
        KV_HEADS,
        HEAD_DIM,
    );
    let mut contiguous_cache = backend.prepare_paged_kv(0, storage)?;
    let mut scattered_cache = backend.prepare_paged_kv(1, storage)?;
    let keys = copy(&backend, &pattern(CONTEXT * KV_HEADS * HEAD_DIM, 31, 15, 32.0)?)?;
    let values = copy(&backend, &pattern(CONTEXT * KV_HEADS * HEAD_DIM, 29, 14, 24.0)?)?;
    let contiguous = block_table(blocks)?;
    let scattered = table(&[4, 6, 5, 7]);
    contiguous_cache.store(
        &KvWritePlan::prefill(Uuid::nil(), 0, &contiguous, 0, CONTEXT)?,
        &keys,
        &values,
    )?;
    scattered_cache.store(
        &KvWritePlan::prefill(Uuid::nil(), 1, &scattered, 0, CONTEXT)?,
        &keys,
        &values,
    )?;
    let query = copy(&backend, &pattern(QUERY_TOKENS * QUERY_HEADS * HEAD_DIM, 23, 11, 16.0)?)?;
    let output_len = QUERY_TOKENS * QUERY_HEADS * HEAD_DIM;
    let mut fmha = backend
        .inner
        .pool
        .allocate::<mircuda::bf16>(&backend.inner.stream, output_len)?;
    let mut paged = backend
        .inner
        .pool
        .allocate::<mircuda::bf16>(&backend.inner.stream, output_len)?;
    let mut reference = backend
        .inner
        .pool
        .allocate::<mircuda::bf16>(&backend.inner.stream, output_len)?;
    let mut attention = backend.prepare_paged_attention_bf16(&contiguous_cache, QUERY_HEADS, 4)?;
    attention.execute_prefill(
        &query,
        &contiguous_cache,
        &contiguous,
        &mut fmha,
        QUERY_TOKENS,
        CONTEXT - QUERY_TOKENS,
        None,
        1.0 / 128.0_f32.sqrt(),
    )?;
    attention.execute_prefill(
        &query,
        &scattered_cache,
        &scattered,
        &mut paged,
        QUERY_TOKENS,
        CONTEXT - QUERY_TOKENS,
        None,
        1.0 / 128.0_f32.sqrt(),
    )?;
    attention.execute_prefill(
        &query,
        &scattered_cache,
        &scattered,
        &mut reference,
        QUERY_TOKENS,
        CONTEXT - QUERY_TOKENS,
        Some(CONTEXT),
        1.0 / 128.0_f32.sqrt(),
    )?;
    let fmha = read(&backend, &fmha)?;
    let paged = read(&backend, &paged)?;
    let reference = read(&backend, &reference)?;
    let direct_error = fmha
        .iter()
        .zip(&reference)
        .map(|(fmha, reference)| (fmha.to_f32() - reference.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    let gather_error = paged
        .iter()
        .zip(reference)
        .map(|(paged, reference)| (paged.to_f32() - reference.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        direct_error <= 0.035_156_25,
        "direct FMHA maximum BF16 difference: {direct_error}"
    );
    assert!(
        gather_error <= 0.035_156_25,
        "gathered FMHA maximum BF16 difference: {gather_error}"
    );
    Ok(())
}

fn block_table(blocks: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in 0..blocks {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(CONTEXT);
    Ok(table)
}

fn table(blocks: &[u32]) -> BlockTable {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in blocks {
        table.push(BlockId(*block));
    }
    table.set_token_len(CONTEXT);
    table
}

fn pattern(
    len: usize,
    modulus: usize,
    midpoint: usize,
    divisor: f32,
) -> Result<Vec<mircuda::bf16>> {
    (0..len)
        .map(|index| {
            let value = i16::try_from(index % modulus)? - i16::try_from(midpoint)?;
            Ok(mircuda::bf16::from_f32(f32::from(value) / divisor))
        })
        .collect()
}
