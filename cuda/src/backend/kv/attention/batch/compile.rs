//! The paged kernels of a batched decode attention plan.

use runtime::kv::KvStorageSpec;

use crate::{
    AttentionExecution, AttentionPlan, AttentionPlanRequest, CudaBackend, Error, Result,
    kernels::{
        BatchedPagedAttention, BatchedSplitPagedAttention, PagedAttentionSpec, ScoreSoftcap,
    },
};

pub(super) fn compile_operations(
    backend: &CudaBackend,
    storage: KvStorageSpec,
    (query_heads, max_blocks, max_batch): (usize, usize, usize),
    score_softcap: Option<ScoreSoftcap>,
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
        score_softcap,
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
