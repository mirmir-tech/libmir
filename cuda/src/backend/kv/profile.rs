use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{
    BlockId, BlockTable, CacheConfig, KvBackendStorage, KvCacheDType, KvStorageSpec, KvWritePlan,
};
use uuid::Uuid;

use super::super::CudaBackend;
use crate::{CudaAttentionPolicy, CudaConfig, CudaPlanningPolicy, Result};

const BLOCK_SIZE: usize = 16;
const KV_HEADS: usize = 8;
const QUERY_HEADS: usize = 16;
const HEAD_DIM: usize = 256;
const ITERATIONS: usize = 200;

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn profile_long_context_paged_attention() -> Result<()> {
    if std::env::var_os("LIBMIR_CUDA_PROFILE_KV_LONG").is_none() {
        return Ok(());
    }
    let backend = CudaBackend::new(CudaConfig::default())?;
    for tokens in [128, 256, 512, 1_024, 2_048, 4_096] {
        let (direct, split) = run(
            &backend,
            KvCacheDType::BFloat16,
            tokens,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            ITERATIONS,
        )?;
        eprintln!(
            "paged attention ({tokens} tokens): direct {direct:.3} ms, split-KV {split:.3} ms, speedup {:.3}x",
            direct / split
        );
    }
    for partition_tokens in [64, 128, 192, 256, 384, 512] {
        let backend = CudaBackend::new(CudaConfig {
            planning: CudaPlanningPolicy {
                attention: CudaAttentionPolicy::SplitKv {
                    partition_tokens,
                    threshold_tokens: partition_tokens + 1,
                },
                ..CudaPlanningPolicy::default()
            },
            ..CudaConfig::default()
        })?;
        for tokens in [128, 256, 576, 1_024, 2_048, 4_096] {
            let (_, split) = run(
                &backend,
                KvCacheDType::BFloat16,
                tokens,
                QUERY_HEADS,
                KV_HEADS,
                HEAD_DIM,
                ITERATIONS,
            )?;
            eprintln!("split-KV partition {partition_tokens} at {tokens} tokens: {split:.3} ms");
        }
    }
    Ok(())
}

#[test]
#[allow(clippy::print_stderr)]
fn profile_gqa_long_context_paged_attention() -> Result<()> {
    if std::env::var_os("LIBMIR_CUDA_PROFILE_KV_GQA_LONG").is_none() {
        return Ok(());
    }
    let tokens = environment_usize("LIBMIR_CUDA_PROFILE_KV_TOKENS", 100_000)?;
    let query_heads = environment_usize("LIBMIR_CUDA_PROFILE_KV_QUERY_HEADS", 64)?;
    let kv_heads = environment_usize("LIBMIR_CUDA_PROFILE_KV_HEADS", 8)?;
    let head_dim = environment_usize("LIBMIR_CUDA_PROFILE_KV_HEAD_DIM", 64)?;
    let iterations = environment_usize("LIBMIR_CUDA_PROFILE_KV_ITERATIONS", 20)?;
    let attention = std::env::var_os("LIBMIR_CUDA_PROFILE_KV_PARTITION")
        .map(|_| environment_usize("LIBMIR_CUDA_PROFILE_KV_PARTITION", 64))
        .transpose()?
        .map_or(CudaAttentionPolicy::Auto, |partition_tokens| CudaAttentionPolicy::SplitKv {
            partition_tokens,
            threshold_tokens: partition_tokens + 1,
        });
    let backend = CudaBackend::new(CudaConfig {
        planning: CudaPlanningPolicy {
            attention,
            ..CudaPlanningPolicy::default()
        },
        ..CudaConfig::default()
    })?;
    let (direct, split) = run(
        &backend,
        KvCacheDType::BFloat16,
        tokens,
        query_heads,
        kv_heads,
        head_dim,
        iterations,
    )?;
    eprintln!(
        "GQA {query_heads}/{kv_heads}x{head_dim} at {tokens} tokens: \
         direct {direct:.3} ms, split-KV {split:.3} ms, speedup {:.3}x",
        direct / split
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run(
    backend: &CudaBackend,
    dtype: KvCacheDType,
    tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    iterations: usize,
) -> Result<(f32, f32)> {
    let blocks = tokens.div_ceil(BLOCK_SIZE);
    let spec = KvStorageSpec::new(
        CacheConfig {
            block_size: BLOCK_SIZE,
            block_count: u32::try_from(blocks)?,
            dtype,
        },
        kv_heads,
        head_dim,
    );
    let mut cache = backend.prepare_paged_kv(0, spec)?;
    let width = tokens * kv_heads * head_dim;
    let keys = patterned_buffer(backend, width, 0x9e37_79b9)?;
    let values = patterned_buffer(backend, width, 0x243f_6a88)?;
    let table = block_table(tokens)?;
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, tokens)?;
    cache.store(&plan, &keys, &values)?;
    let query = patterned_buffer(backend, query_heads * head_dim, 0xb7e1_5163)?;
    let mut output = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, query_heads * head_dim)?;
    let mut attention = backend.prepare_paged_attention_bf16(&cache, query_heads, blocks)?;
    execute(&mut attention, &query, &cache, &table, &mut output)?;
    backend.inner.stream.synchronize()?;
    let direct =
        measure(backend, &mut attention, &query, &cache, &table, &mut output, false, iterations)?;
    let split =
        measure(backend, &mut attention, &query, &cache, &table, &mut output, true, iterations)?;
    Ok((direct, split))
}

fn patterned_buffer(backend: &CudaBackend, len: usize, seed: u32) -> Result<DeviceBuffer<bf16>> {
    let mut state = seed;
    let values = std::iter::repeat_with(|| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let sign = if state & 0x80 == 0 {
            0
        } else {
            0x8000
        };
        let mantissa = u16::from(state.to_le_bytes()[0] & 0x7f);
        bf16::from_bits(sign | 0x3f00 | mantissa)
    })
    .take(len)
    .collect::<Vec<_>>();
    let mut host = backend.inner.context.allocate_pinned::<bf16>(len)?;
    host.copy_from_slice(&values)?;
    let mut device = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, len)?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.inner.stream.synchronize()?;
    Ok(device)
}

#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn measure(
    backend: &CudaBackend,
    attention: &mut super::PagedAttentionBf16,
    query: &DeviceBuffer<bf16>,
    cache: &super::PagedKvCache,
    table: &BlockTable,
    output: &mut DeviceBuffer<bf16>,
    split: bool,
    iterations: usize,
) -> Result<f32> {
    let started = backend.inner.context.create_event(true)?;
    let completed = backend.inner.context.create_event(true)?;
    started.record(&backend.inner.stream)?;
    for _ in 0..iterations {
        if split {
            attention.execute_split(query, cache, table, output, None, 0.0625)?;
        } else {
            attention.execute_direct(query, cache, table, output, None, 0.0625)?;
        }
    }
    completed.record(&backend.inner.stream)?;
    completed.synchronize()?;
    Ok(started.elapsed_ms(&completed)? / iterations as f32)
}

fn execute(
    attention: &mut super::PagedAttentionBf16,
    query: &DeviceBuffer<bf16>,
    cache: &super::PagedKvCache,
    table: &BlockTable,
    output: &mut DeviceBuffer<bf16>,
) -> Result<()> {
    attention.execute(query, cache, table, output, None, 0.0625)
}

fn block_table(tokens: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(BLOCK_SIZE);
    for block in 0..tokens.div_ceil(BLOCK_SIZE) {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(tokens);
    Ok(table)
}

fn environment_usize(name: &str, default: usize) -> Result<usize> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(default);
    };
    value
        .parse()
        .or(Err(crate::Error::InvalidPagedKv("invalid profiling environment value")))
}
