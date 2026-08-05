use mircuda::{DeviceBuffer, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file};
use runtime::kv::KvStorageSpec;

use crate::{
    AttentionExecution, AttentionPlanRequest, CudaBackend, Error, PlanSource, Result,
    backend::{AttentionFamily, AttentionProfileRequest},
    kernels::{PagedAttentionSpec, SplitAttentionWorkspace, SplitPagedAttention},
};

mod autotune;
mod batch;

pub use batch::ClampedRoutedBatchSplitDecode;

cuda_export!(SinkMergeKernel = "libmir_cuda_clamped_routed_sink_merge_bf16"(
    partial_values: &DeviceBuffer<f32>, partial_maxima: &DeviceBuffer<f32>,
    partial_denominators: &DeviceBuffer<f32>, sinks: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>, query_heads: u32, head_dim: u32,
    active_partitions: u32, max_partitions: u32,
));

#[derive(Debug)]
pub struct ClampedRoutedSplitDecode {
    operation: SplitPagedAttention,
    workspace: SplitAttentionWorkspace,
    merge: TypedKernel<SinkMergeKernel>,
    threshold: usize,
    partition_tokens: usize,
    backend: CudaBackend,
    request: AttentionProfileRequest,
    fallback: AttentionExecution,
    profile_allowed: bool,
    tuning_complete: bool,
}

impl ClampedRoutedSplitDecode {
    pub(crate) fn compile(
        backend: &CudaBackend,
        storage: KvStorageSpec,
        query_heads: usize,
        max_blocks: usize,
    ) -> Result<Option<Self>> {
        let max_context_tokens = max_blocks
            .checked_mul(storage.cache.block_size)
            .ok_or(Error::InvalidExecutionPlan("split attention context overflow"))?;
        let plan = backend.execution_planner().plan_attention(AttentionPlanRequest {
            max_context_tokens,
            query_heads,
            kv_heads: storage.kv_heads,
            head_dim: storage.key_head_dim,
            value_head_dim: storage.value_head_dim,
        })?;
        if plan.execution() == AttentionExecution::Direct
            && plan.source() == PlanSource::ExplicitPolicy
        {
            return Ok(None);
        }
        let (partition_tokens, threshold_tokens) = match plan.execution() {
            AttentionExecution::Direct => (256, max_context_tokens + 1),
            AttentionExecution::SplitKv { partition_tokens, threshold_tokens } => {
                (partition_tokens, threshold_tokens)
            },
        };
        let attention_request = AttentionPlanRequest {
            max_context_tokens,
            query_heads,
            kv_heads: storage.kv_heads,
            head_dim: storage.key_head_dim,
            value_head_dim: storage.value_head_dim,
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
        let operation = SplitPagedAttention::compile(backend.compiler(), spec, partition_tokens)?;
        let (values, statistics) = operation.workspace_lengths();
        let workspace = SplitAttentionWorkspace::new(
            backend.pool().allocate(backend.stream(), values)?,
            backend.pool().allocate(backend.stream(), statistics)?,
            backend.pool().allocate(backend.stream(), statistics)?,
        );
        let module = backend.compiler().compile(
            cuda_kernel_file!("../../../../../kernels/clamped_routed_attention_bf16.cu"),
            &super::super::super::paged::compile_options(storage.cache.dtype)?,
        )?;
        Ok(Some(Self {
            operation,
            workspace,
            merge: module.kernel()?,
            threshold: threshold_tokens,
            partition_tokens,
            backend: backend.clone(),
            request: AttentionProfileRequest {
                family: AttentionFamily::ClampedSink,
                plan: attention_request,
                batch_rows: 1,
                block_size: storage.cache.block_size,
                dtype: storage.cache.dtype,
                window_tokens: None,
            },
            fallback: plan.execution(),
            profile_allowed: plan.source() != PlanSource::ExplicitPolicy,
            tuning_complete: false,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute(
        &mut self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        table: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        tokens: usize,
        blocks: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<bool> {
        let visible = window.map_or(tokens, |limit| tokens.min(limit));
        if visible < self.threshold {
            return Ok(false);
        }
        let active = self.operation.execute_partitions(
            stream,
            query,
            key_pages,
            value_pages,
            table,
            &mut self.workspace,
            output,
            tokens,
            blocks,
            window,
            scale,
        )?;
        let query_heads = sinks.len();
        let head_dim = output.len() / query_heads;
        self.merge.launch(
            stream,
            self.operation.configs(active)?.merge,
            (
                &self.workspace.values,
                &self.workspace.maxima,
                &self.workspace.denominators,
                sinks,
                output,
                u32::try_from(query_heads)?,
                u32::try_from(head_dim)?,
                u32::try_from(active)?,
                u32::try_from(self.operation.max_partitions())?,
            ),
        )?;
        Ok(true)
    }
}
