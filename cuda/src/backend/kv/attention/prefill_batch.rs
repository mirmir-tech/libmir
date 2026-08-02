use mircuda::{DeviceBuffer, Stream, bf16};

use super::super::{PagedKvCache, PagedPrefillBatch};
use crate::{
    CudaBackend, Error, Result,
    kernels::{BatchedPagedPrefillAttention, PagedAttentionSpec},
};

/// Variable-length causal prefill attention over packed query rows.
#[derive(Debug)]
pub struct BatchedPrefillPagedAttentionBf16 {
    operation: BatchedPagedPrefillAttention,
    stream: Stream,
    storage: runtime::kv::KvStorageSpec,
}

impl BatchedPrefillPagedAttentionBf16 {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        cache: &PagedKvCache,
        query_heads: usize,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<Self> {
        let storage = cache.storage_spec();
        let spec = PagedAttentionSpec {
            block_size: storage.cache.block_size,
            max_blocks,
            query_heads,
            kv_heads: storage.kv_heads,
            head_dim: storage.key_head_dim,
            value_head_dim: storage.value_head_dim,
            dtype: storage.cache.dtype,
        };
        Ok(Self {
            operation: BatchedPagedPrefillAttention::compile(
                &backend.inner.compiler,
                spec,
                max_batch,
            )?,
            stream: backend.inner.stream.clone(),
            storage,
        })
    }

    pub fn execute(
        &self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        if cache.storage_spec() != self.storage
            || batch.cache_config() != self.storage.cache
            || batch.active() == 0
        {
            return Err(Error::InvalidPagedKv(
                "batched prefill attention received another KV geometry",
            ));
        }
        self.operation.execute(
            &self.stream,
            query,
            cache.key_pages(),
            cache.value_pages(),
            batch.tables(),
            batch.block_counts(),
            batch.request_indices(),
            batch.positions(),
            output,
            batch.tokens(),
            batch.active(),
            window,
            scale,
        )
    }
}
