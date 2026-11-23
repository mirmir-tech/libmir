//! Llama 4's chunked attention: a query sees the keys since its chunk began.

use models::semantic::AttentionWindow;
use runtime::kv::{CacheConfig, KvBackendStorage, KvCacheDType, KvStorageSpec, KvWritePlan};
use uuid::Uuid;

use super::{
    super::super::CudaBackend, assert_output, assert_token_output, bf16s, block_table, copy, read,
    reference,
};
use crate::CudaConfig;

const CHUNK: Option<AttentionWindow> = Some(AttentionWindow::Chunked { chunk: 2 });

#[test]
fn chunked_queries_see_only_their_chunk() -> Result<(), Box<dyn std::error::Error>> {
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
    let mut attention = backend.prepare_paged_attention_bf16(&cache, 2, 2)?;

    // Decode at position 2 starts the second chunk: only its own key.
    let query = copy(&backend, &bf16s(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    attention.execute(&query, &cache, &table, &mut output, CHUNK, 0.5)?;
    assert_output(&read(&backend, &output)?, &reference(&[0.5], Some(2)));

    // Prefill of positions 1 and 2: the first ends chunk zero, the second
    // starts chunk one.
    let query = copy(&backend, &bf16s(&[1.0, 0.0, 0.0, 0.0].repeat(4)))?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 16)?;
    attention.execute_prefill(&query, &cache, &table, &mut output, 2, 1, CHUNK, 0.5)?;
    let actual = read(&backend, &output)?;
    assert_token_output(&actual, 0, &reference(&[0.5, 0.0], None));
    assert_token_output(&actual, 1, &reference(&[0.5], Some(2)));
    Ok(())
}
