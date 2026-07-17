use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::kv::KvStorageSpec;

use super::super::{PagedDecodeBatch, PagedKvCache};
use crate::{
    CudaBackend, Error, Result,
    kernels::{BatchedPagedAttention, PagedAttentionSpec},
};

/// Allocation-free decode attention over multiple independent block tables.
#[derive(Debug)]
pub struct BatchedPagedAttentionBf16 {
    operation: BatchedPagedAttention,
    stream: Stream,
    max_batch: usize,
    max_blocks: usize,
    storage: KvStorageSpec,
}

impl BatchedPagedAttentionBf16 {
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
            operation: BatchedPagedAttention::compile(&backend.inner.compiler, spec, max_batch)?,
            stream: backend.inner.stream.clone(),
            max_batch,
            max_blocks,
            storage,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        if cache.storage_spec() != self.storage
            || batch.cache_config() != self.storage.cache
            || batch.max_blocks() != self.max_blocks
            || batch.active() > self.max_batch
        {
            return Err(Error::InvalidPagedKv("batched attention metadata geometry differs"));
        }
        self.operation.execute(
            &self.stream,
            query,
            cache.key_pages(),
            cache.value_pages(),
            batch.tables(),
            batch.token_counts(),
            batch.block_counts(),
            output,
            batch.active(),
            window,
            scale,
        )
    }
}
