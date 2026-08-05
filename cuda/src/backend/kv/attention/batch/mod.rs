use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::kv::KvStorageSpec;

use super::super::{PagedDecodeBatch, PagedKvCache};
use crate::{
    AttentionExecution, AttentionPlan, AttentionPlanRequest, CudaBackend, Error, PlanSource,
    Result,
    kernels::{
        BatchedPagedAttention, BatchedSplitAttentionWorkspace, BatchedSplitPagedAttention,
        PagedAttentionSpec,
    },
};

mod fmha;
mod tuning;

use fmha::PagedFmhaDecode;

#[derive(Debug)]
pub struct BatchedPagedAttentionBf16 {
    operation: BatchedPagedAttention,
    split: BatchedSplitPagedAttention,
    split_workspace: BatchedSplitAttentionWorkspace,
    fmha: Option<PagedFmhaDecode>,
    stream: Stream,
    max_batch: usize,
    max_blocks: usize,
    split_threshold: usize,
    plan_request: AttentionPlanRequest,
    fallback_execution: AttentionExecution,
    profile_allowed: bool,
    tuning_complete: bool,
    backend: CudaBackend,
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
        Self::workspace_lengths_for_storage(
            backend,
            cache.storage_spec(),
            query_heads,
            max_blocks,
            max_batch,
        )
    }

    pub(in crate::backend) fn workspace_lengths_for_storage(
        backend: &CudaBackend,
        storage: KvStorageSpec,
        query_heads: usize,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<(usize, usize)> {
        let (_, split, _, _) =
            compile_operations(backend, storage, query_heads, max_blocks, max_batch)?;
        let partition = super::autotune::candidate_partitions(split.partition_tokens())
            .into_iter()
            .min()
            .ok_or(Error::InvalidExecutionPlan("attention tuner has no partitions"))?;
        let split = BatchedSplitPagedAttention::compile(
            &backend.inner.compiler,
            split.spec(),
            max_batch,
            partition,
        )?;
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
        let (operation, split, plan_request, plan) =
            compile_operations(backend, storage, query_heads, max_blocks, max_batch)?;
        let split_threshold = threshold(plan.execution(), plan_request.max_context_tokens);
        let split_workspace = workspace.map_or_else(|| allocate_workspace(backend, &split), Ok)?;
        let fmha = PagedFmhaDecode::prepare(backend, storage, query_heads, max_blocks, max_batch)?;
        Ok(Self {
            operation,
            split,
            split_workspace,
            fmha,
            stream: backend.inner.stream.clone(),
            max_batch,
            max_blocks,
            split_threshold,
            plan_request,
            fallback_execution: plan.execution(),
            profile_allowed: plan.source() != PlanSource::ExplicitPolicy,
            tuning_complete: false,
            backend: backend.clone(),
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
        if window.is_none()
            && let Some(fmha) = &mut self.fmha
        {
            return fmha.execute(query, cache, batch, output, scale);
        }
        self.ensure_tuned(query, cache, batch, output, window, scale);
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

    pub(crate) fn capture_partitions(&self, batch: &PagedDecodeBatch) -> usize {
        if self.fmha.is_some() {
            return PagedFmhaDecode::capture_key(batch);
        }
        batch
            .maximum_tokens()
            .div_ceil(self.split.partition_tokens())
            .min(self.split.max_partitions())
    }
}

fn compile_operations(
    backend: &CudaBackend,
    storage: KvStorageSpec,
    query_heads: usize,
    max_blocks: usize,
    max_batch: usize,
) -> Result<(
    BatchedPagedAttention,
    BatchedSplitPagedAttention,
    AttentionPlanRequest,
    AttentionPlan,
)> {
    let max_context_tokens = max_blocks
        .checked_mul(storage.cache.block_size)
        .ok_or(Error::InvalidExecutionPlan("attention context capacity overflow"))?;
    let request = AttentionPlanRequest {
        max_context_tokens,
        query_heads,
        kv_heads: storage.kv_heads,
        head_dim: storage.key_head_dim,
        value_head_dim: storage.value_head_dim,
    };
    let plan = backend.execution_planner().plan_attention(request)?;
    let partition_tokens = match plan.execution() {
        AttentionExecution::Direct => 256,
        AttentionExecution::SplitKv { partition_tokens, .. } => partition_tokens,
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
        request,
        plan,
    ))
}

const fn threshold(execution: AttentionExecution, max_context_tokens: usize) -> usize {
    match execution {
        AttentionExecution::Direct => max_context_tokens + 1,
        AttentionExecution::SplitKv { threshold_tokens, .. } => threshold_tokens,
    }
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
