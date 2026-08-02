use mircuda::{
    DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};
use runtime::kv::KvStorageSpec;

use crate::{
    AttentionExecution, AttentionPlanRequest, CudaBackend, Error, PagedPrefillBatch, PlanSource,
    Result,
    kernels::{
        BatchedSplitAttentionWorkspace, BatchedSplitPagedAttention, PagedAttentionSpec, paged,
    },
};

cuda_export!(BatchSinkMergeKernel = "libmir_cuda_clamped_routed_sink_batch_merge_bf16"(
    partial_values: &DeviceBuffer<f32>, partial_maxima: &DeviceBuffer<f32>,
    partial_denominators: &DeviceBuffer<f32>, token_counts: &DeviceBuffer<u32>,
    sinks: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    batch_size: u32, query_heads: u32, head_dim: u32, window: u32,
    partition_tokens: u32, max_partitions: u32, minimum_tokens: u32,
));

#[derive(Debug)]
pub struct ClampedRoutedBatchSplitDecode {
    operation: BatchedSplitPagedAttention,
    workspace: BatchedSplitAttentionWorkspace,
    merge: TypedKernel<BatchSinkMergeKernel>,
    threshold: usize,
}

impl ClampedRoutedBatchSplitDecode {
    pub(crate) fn compile(
        backend: &CudaBackend,
        storage: KvStorageSpec,
        query_heads: usize,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<Option<Self>> {
        let max_context_tokens = max_blocks
            .checked_mul(storage.cache.block_size)
            .ok_or(Error::InvalidExecutionPlan("batched sink attention context overflow"))?;
        let request = AttentionPlanRequest {
            max_context_tokens,
            query_heads,
            kv_heads: storage.kv_heads,
            head_dim: storage.key_head_dim,
            value_head_dim: storage.value_head_dim,
        };
        let plan = backend.execution_planner().plan_attention(request)?;
        if plan.execution() == AttentionExecution::Direct
            && plan.source() == PlanSource::ExplicitPolicy
        {
            return Ok(None);
        }
        let (partition_tokens, threshold) = match plan.execution() {
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
        let operation = BatchedSplitPagedAttention::compile(
            backend.compiler(),
            spec,
            max_batch,
            partition_tokens,
        )?;
        let (values, statistics) = operation.workspace_lengths();
        let workspace = BatchedSplitAttentionWorkspace::new(
            backend.pool().allocate(backend.stream(), values)?,
            backend.pool().allocate(backend.stream(), statistics)?,
            backend.pool().allocate(backend.stream(), statistics)?,
        );
        let module = backend.compiler().compile(
            cuda_kernel_file!("../../../../../kernels/clamped_routed_attention_bf16.cu"),
            &paged::compile_options(storage.cache.dtype)?,
        )?;
        Ok(Some(Self {
            operation,
            workspace,
            merge: module.kernel()?,
            threshold,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute(
        &mut self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        batch: &PagedPrefillBatch,
        sinks: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) -> Result<bool> {
        if window.is_some() || self.capture_partitions(batch) == 0 {
            return Ok(false);
        }
        self.operation.execute_partitions(
            stream,
            query,
            key_pages,
            value_pages,
            batch.tables(),
            batch.token_counts(),
            batch.block_counts(),
            &mut self.workspace,
            output,
            batch.active(),
            None,
            scale,
            self.threshold,
            batch.max_context_tokens(),
        )?;
        let query_heads = sinks.len();
        let head_dim = output.len() / (batch.active() * query_heads);
        self.merge.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(query_heads)?, u32::try_from(batch.active())?, 1),
                block: (128, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                &self.workspace.values,
                &self.workspace.maxima,
                &self.workspace.denominators,
                batch.token_counts(),
                sinks,
                output,
                u32::try_from(batch.active())?,
                u32::try_from(query_heads)?,
                u32::try_from(head_dim)?,
                0,
                u32::try_from(self.operation.partition_tokens())?,
                u32::try_from(self.operation.max_partitions())?,
                u32::try_from(self.threshold)?,
            ),
        )?;
        Ok(true)
    }

    pub(crate) fn capture_partitions(&self, batch: &PagedPrefillBatch) -> usize {
        if batch.max_query_tokens() != 1
            || batch.tokens() != batch.active()
            || batch.max_context_tokens() < self.threshold
        {
            return 0;
        }
        batch
            .max_context_tokens()
            .div_ceil(self.operation.partition_tokens())
            .min(self.operation.max_partitions())
    }
}
