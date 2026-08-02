use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::kv::KvStorageSpec;

use super::super::{PagedDecodeBatch, PagedKvCache};
use crate::{
    AttentionExecution, AttentionPlanRequest, CudaBackend, Error, Result,
    kernels::{
        BatchedPagedAttention, BatchedSplitAttentionWorkspace, BatchedSplitPagedAttention,
        PagedAttentionSpec,
    },
};

/// Allocation-free decode attention over multiple independent block tables.
#[derive(Debug)]
pub struct BatchedPagedAttentionBf16 {
    operation: BatchedPagedAttention,
    split: BatchedSplitPagedAttention,
    split_workspace: BatchedSplitAttentionWorkspace,
    stream: Stream,
    max_batch: usize,
    max_blocks: usize,
    split_threshold: usize,
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
        Self::new_with_workspace(backend, cache, query_heads, max_blocks, max_batch, None)
    }

    pub(in crate::backend) fn workspace_lengths(
        backend: &CudaBackend,
        cache: &PagedKvCache,
        query_heads: usize,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<(usize, usize)> {
        let (_, split, _) = compile_operations(backend, cache, query_heads, max_blocks, max_batch)?;
        Ok(split.workspace_lengths())
    }

    pub(in crate::backend) fn new_with_workspace(
        backend: &CudaBackend,
        cache: &PagedKvCache,
        query_heads: usize,
        max_blocks: usize,
        max_batch: usize,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<Self> {
        let storage = cache.storage_spec();
        let (operation, split, split_threshold) =
            compile_operations(backend, cache, query_heads, max_blocks, max_batch)?;
        let split_workspace = workspace.map_or_else(|| allocate_workspace(backend, &split), Ok)?;
        Ok(Self {
            operation,
            split,
            split_workspace,
            stream: backend.inner.stream.clone(),
            max_batch,
            max_blocks,
            split_threshold,
            storage,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
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
            self.split_threshold,
        )?;
        self.split.execute(
            &self.stream,
            query,
            cache.key_pages(),
            cache.value_pages(),
            batch.tables(),
            batch.token_counts(),
            batch.block_counts(),
            &mut self.split_workspace,
            output,
            batch.active(),
            window,
            scale,
            self.split_threshold,
            batch.maximum_tokens(),
        )
    }
}

fn compile_operations(
    backend: &CudaBackend,
    cache: &PagedKvCache,
    query_heads: usize,
    max_blocks: usize,
    max_batch: usize,
) -> Result<(BatchedPagedAttention, BatchedSplitPagedAttention, usize)> {
    let storage = cache.storage_spec();
    let max_context_tokens = max_blocks
        .checked_mul(storage.cache.block_size)
        .ok_or(Error::InvalidExecutionPlan("attention context capacity overflow"))?;
    let plan = backend.execution_planner().plan_attention(AttentionPlanRequest {
        max_context_tokens,
        query_heads,
        kv_heads: storage.kv_heads,
        head_dim: storage.key_head_dim,
        value_head_dim: storage.value_head_dim,
    })?;
    let (partition_tokens, split_threshold) = match plan.execution() {
        AttentionExecution::Direct => (256, max_context_tokens + 1),
        AttentionExecution::SplitKv { partition_tokens, threshold_tokens } => {
            (partition_tokens, threshold_tokens)
        },
    };
    let spec = PagedAttentionSpec {
        block_size: storage.cache.block_size,
        max_blocks,
        query_heads,
        kv_heads: storage.kv_heads,
        head_dim: storage.key_head_dim,
        value_head_dim: storage.value_head_dim,
        dtype: storage.cache.dtype,
    };
    Ok((
        BatchedPagedAttention::compile(&backend.inner.compiler, spec, max_batch)?,
        BatchedSplitPagedAttention::compile(
            &backend.inner.compiler,
            spec,
            max_batch,
            partition_tokens,
        )?,
        split_threshold,
    ))
}

fn allocate_workspace(
    backend: &CudaBackend,
    split: &BatchedSplitPagedAttention,
) -> Result<BatchedSplitAttentionWorkspace> {
    let (values, statistics) = split.workspace_lengths();
    Ok(BatchedSplitAttentionWorkspace::new(
        backend.inner.pool.allocate(&backend.inner.stream, values)?,
        backend.inner.pool.allocate(&backend.inner.stream, statistics)?,
        backend.inner.pool.allocate(&backend.inner.stream, statistics)?,
    ))
}
