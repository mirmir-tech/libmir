use mircuda::bf16;
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
const ITERATIONS_F32: f32 = 200.0;

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn profile_long_context_paged_attention() -> Result<()> {
    if std::env::var_os("LIBMIR_CUDA_PROFILE_KV_LONG").is_none() {
        return Ok(());
    }
    let backend = CudaBackend::new(CudaConfig::default())?;
    for tokens in [128, 256, 512, 1_024, 2_048, 4_096] {
        let (direct, split) = run(&backend, KvCacheDType::BFloat16, tokens)?;
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
            let (_, split) = run(&backend, KvCacheDType::BFloat16, tokens)?;
            eprintln!("split-KV partition {partition_tokens} at {tokens} tokens: {split:.3} ms");
        }
    }
    Ok(())
}

fn run(backend: &CudaBackend, dtype: KvCacheDType, tokens: usize) -> Result<(f32, f32)> {
    let blocks = tokens.div_ceil(BLOCK_SIZE);
    let spec = KvStorageSpec::new(
        CacheConfig {
            block_size: BLOCK_SIZE,
            block_count: u32::try_from(blocks)?,
            dtype,
        },
        KV_HEADS,
        HEAD_DIM,
    );
    let mut cache = backend.prepare_paged_kv(0, spec)?;
    let width = tokens * KV_HEADS * HEAD_DIM;
    let keys = backend.inner.pool.allocate_zeroed::<bf16>(&backend.inner.stream, width)?;
    let values = backend.inner.pool.allocate_zeroed::<bf16>(&backend.inner.stream, width)?;
    let table = block_table(tokens)?;
    let plan = KvWritePlan::prefill(Uuid::nil(), 0, &table, 0, tokens)?;
    cache.store(&plan, &keys, &values)?;
    let query = backend
        .inner
        .pool
        .allocate_zeroed::<bf16>(&backend.inner.stream, QUERY_HEADS * HEAD_DIM)?;
    let mut output = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, QUERY_HEADS * HEAD_DIM)?;
    let mut attention = backend.prepare_paged_attention_bf16(&cache, QUERY_HEADS, blocks)?;
    execute(&mut attention, &query, &cache, &table, &mut output)?;
    backend.inner.stream.synchronize()?;
    let direct = measure(backend, &mut attention, &query, &cache, &table, &mut output, false)?;
    let split = measure(backend, &mut attention, &query, &cache, &table, &mut output, true)?;
    Ok((direct, split))
}

#[allow(clippy::too_many_arguments)]
fn measure(
    backend: &CudaBackend,
    attention: &mut super::PagedAttentionBf16,
    query: &mircuda::DeviceBuffer<bf16>,
    cache: &super::PagedKvCache,
    table: &BlockTable,
    output: &mut mircuda::DeviceBuffer<bf16>,
    split: bool,
) -> Result<f32> {
    let started = backend.inner.context.create_event(true)?;
    let completed = backend.inner.context.create_event(true)?;
    started.record(&backend.inner.stream)?;
    for _ in 0..ITERATIONS {
        if split {
            attention.execute_split(query, cache, table, output, None, 0.0625)?;
        } else {
            attention.execute_direct(query, cache, table, output, None, 0.0625)?;
        }
    }
    completed.record(&backend.inner.stream)?;
    completed.synchronize()?;
    Ok(started.elapsed_ms(&completed)? / ITERATIONS_F32)
}

fn execute(
    attention: &mut super::PagedAttentionBf16,
    query: &mircuda::DeviceBuffer<bf16>,
    cache: &super::PagedKvCache,
    table: &BlockTable,
    output: &mut mircuda::DeviceBuffer<bf16>,
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
