use ::runtime::kv::{BlockTable, KvStorageSpec};
use mircuda::{DeviceBuffer, KernelNode, PinnedBuffer, Stream, TypedKernel};

use super::{
    super::{AttentionExecution, AttentionPlanRequest, CudaBackend},
    PagedKvCache,
};
use crate::{
    Error, Result,
    kernels::{
        AttentionKernel, PagedAttention, PagedAttentionSpec, PagedPrefillAttention,
        SplitAttentionConfigs, SplitAttentionKernels, SplitAttentionNodes, SplitAttentionWorkspace,
        SplitPagedAttention,
    },
};

mod batch;
mod execution;

pub use batch::BatchedPagedAttentionBf16;

/// Allocation-free decode attention over runtime-managed physical K/V pages.
#[derive(Debug)]
pub struct PagedAttentionBf16 {
    operation: PagedAttention,
    split: SplitPagedAttention,
    split_workspace: SplitAttentionWorkspace,
    split_threshold: u32,
    prefill: PagedPrefillAttention,
    stream: Stream,
    table_device: DeviceBuffer<u32>,
    table_staging: PinnedBuffer<u32>,
    table_snapshot: Vec<u32>,
    table_len: usize,
    storage: KvStorageSpec,
}

#[derive(Clone, Copy)]
pub(in crate::backend) struct CapturedPagedAttentionNodes {
    pub(in crate::backend) direct: KernelNode<AttentionKernel>,
    pub(in crate::backend) split: SplitAttentionNodes,
}

pub(in crate::backend) struct CapturedPagedAttentionKernels {
    pub(in crate::backend) direct: TypedKernel<AttentionKernel>,
    pub(in crate::backend) split: SplitAttentionKernels,
}

impl PagedAttentionBf16 {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        cache: &PagedKvCache,
        query_heads: usize,
        max_blocks: usize,
    ) -> Result<Self> {
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
        let operation = PagedAttention::compile(&backend.inner.compiler, spec)?;
        let split = SplitPagedAttention::compile(&backend.inner.compiler, spec, partition_tokens)?;
        let (partial_values, partial_statistics) = split.workspace_lengths();
        let split_workspace = SplitAttentionWorkspace::new(
            backend.inner.pool.allocate::<f32>(&backend.inner.stream, partial_values)?,
            backend.inner.pool.allocate::<f32>(&backend.inner.stream, partial_statistics)?,
            backend.inner.pool.allocate::<f32>(&backend.inner.stream, partial_statistics)?,
        );
        let prefill = PagedPrefillAttention::compile(&backend.inner.compiler, spec)?;
        let table_device = backend.inner.pool.allocate::<u32>(&backend.inner.stream, max_blocks)?;
        let table_staging = backend.inner.context.allocate_pinned::<u32>(max_blocks)?;
        Ok(Self {
            operation,
            split,
            split_workspace,
            split_threshold: u32::try_from(split_threshold)?,
            prefill,
            stream: backend.inner.stream.clone(),
            table_device,
            table_staging,
            table_snapshot: vec![u32::MAX; max_blocks],
            table_len: 0,
            storage,
        })
    }

    pub(crate) fn prepare_table(&mut self, table: &BlockTable) -> Result<()> {
        self.validate_table(table)?;
        self.update_table(table)
    }

    pub(crate) const fn table_device(&self) -> &DeviceBuffer<u32> {
        &self.table_device
    }

    pub(in crate::backend) fn captured_kernels(&self) -> CapturedPagedAttentionKernels {
        CapturedPagedAttentionKernels {
            direct: self.operation.kernel(),
            split: self.split.kernels(),
        }
    }

    pub(in crate::backend) fn captured_configs(
        &self,
        token_count: u32,
        window: u32,
    ) -> Result<SplitAttentionConfigs> {
        let active = self.split.active_partitions(token_count, window)?;
        self.split.configs(usize::try_from(active)?)
    }

    pub(crate) const fn split_threshold(&self) -> u32 {
        self.split_threshold
    }

    fn validate(&self, cache: &PagedKvCache, table: &BlockTable) -> Result<()> {
        if cache.storage_spec() != self.storage {
            return Err(Error::InvalidPagedKv("attention received another KV arena geometry"));
        }
        self.validate_table(table)
    }

    fn validate_table(&self, table: &BlockTable) -> Result<()> {
        if table.block_size() != Some(self.storage.cache.block_size) {
            return Err(Error::InvalidPagedKv("block table uses another KV block size"));
        }
        let capacity = table
            .blocks()
            .len()
            .checked_mul(self.storage.cache.block_size)
            .ok_or(Error::InvalidPagedKv("block table capacity overflow"))?;
        if table.token_len() > capacity {
            return Err(Error::InvalidPagedKv("block table cannot hold its declared tokens"));
        }
        let physical_blocks = self.storage.cache.block_count;
        if table.blocks().iter().any(|block| block.0 >= physical_blocks) {
            return Err(Error::InvalidPagedKv("block table references a missing physical page"));
        }
        Ok(())
    }

    fn update_table(&mut self, table: &BlockTable) -> Result<()> {
        let blocks = table.blocks();
        if blocks.len() > self.table_snapshot.len() {
            return Err(Error::InvalidPagedKv("block table exceeds prepared sequence capacity"));
        }
        let changed = self.table_len != blocks.len()
            || blocks
                .iter()
                .zip(&self.table_snapshot)
                .any(|(block, cached)| block.0 != *cached);
        if !changed {
            return Ok(());
        }
        self.table_snapshot.fill(u32::MAX);
        for (target, block) in self.table_snapshot.iter_mut().zip(blocks) {
            *target = block.0;
        }
        self.table_staging.copy_from_slice(&self.table_snapshot)?;
        self.stream.copy_to_device(&mut self.table_staging, &mut self.table_device)?;
        self.table_len = blocks.len();
        Ok(())
    }
}
