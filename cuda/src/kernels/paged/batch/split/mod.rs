use mircuda::{
    Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::super::{PagedAttentionSpec, compile_options, validate_attention};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

mod partitions;

cuda_export!(BatchSplitKernel = "libmir_cuda_paged_attention_batch_split_bf16"(
    query: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>, block_tables: &DeviceBuffer<u32>,
    token_counts: &DeviceBuffer<u32>, block_counts: &DeviceBuffer<u32>,
    partial_values: &mut DeviceBuffer<f32>, partial_maxima: &mut DeviceBuffer<f32>,
    partial_denominators: &mut DeviceBuffer<f32>, batch_size: u32, max_blocks: u32,
    block_size: u32, query_heads: u32, kv_heads: u32, head_dim: u32,
    value_head_dim: u32, window: u32, scale: f32, partition_tokens: u32,
    launch_partitions: u32, max_partitions: u32, minimum_tokens: u32,
));

cuda_export!(BatchMergeKernel = "libmir_cuda_paged_attention_batch_merge_bf16"(
    partial_values: &DeviceBuffer<f32>, partial_maxima: &DeviceBuffer<f32>,
    partial_denominators: &DeviceBuffer<f32>, token_counts: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, batch_size: u32, query_heads: u32,
    value_head_dim: u32, window: u32, partition_tokens: u32,
    max_partitions: u32, minimum_tokens: u32,
));

#[derive(Clone, Debug)]
pub struct BatchedSplitPagedAttention {
    split: TypedKernel<BatchSplitKernel>,
    merge: TypedKernel<BatchMergeKernel>,
    spec: PagedAttentionSpec,
    max_batch: usize,
    partition_tokens: usize,
    max_partitions: usize,
}

#[derive(Clone, Debug)]
pub struct BatchedSplitAttentionWorkspace {
    pub(crate) values: DeviceBuffer<f32>,
    pub(crate) maxima: DeviceBuffer<f32>,
    pub(crate) denominators: DeviceBuffer<f32>,
}

impl BatchedSplitPagedAttention {
    pub fn compile(
        compiler: &Compiler,
        spec: PagedAttentionSpec,
        max_batch: usize,
        partition_tokens: usize,
    ) -> Result<Self> {
        validate_attention(spec)?;
        if max_batch == 0 || partition_tokens == 0 {
            return Err(Error::InvalidPagedKv("invalid batched split attention geometry"));
        }
        let max_tokens = product(spec.max_blocks, spec.block_size)?;
        let source =
            cuda_kernel_file!("../../../../../kernels/paged_attention_batch_split_bf16.cu");
        let mut options = compile_options(spec.dtype)?;
        options
            .extra_options
            .push(format!("-DLIBMIR_QUERY_GROUP={}", spec.query_heads / spec.kv_heads));
        options
            .extra_options
            .push(format!("-DLIBMIR_VALUE_ITEMS={}", spec.value_head_dim.div_ceil(128)));
        if super::super::split::tensor_queries(spec, spec.query_heads / spec.kv_heads, true) {
            options.extra_options.push("-DLIBMIR_BATCH_SPLIT_GQA_WMMA=1".into());
        }
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            split: module.kernel()?,
            merge: module.kernel()?,
            spec,
            max_batch,
            partition_tokens,
            max_partitions: max_tokens.div_ceil(partition_tokens),
        })
    }

    #[must_use]
    pub const fn workspace_lengths(&self) -> (usize, usize) {
        (
            self.max_batch * self.spec.query_heads * self.max_partitions * self.spec.value_head_dim,
            self.max_batch * self.spec.query_heads * self.max_partitions,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        block_tables: &DeviceBuffer<u32>,
        token_counts: &DeviceBuffer<u32>,
        block_counts: &DeviceBuffer<u32>,
        workspace: &mut BatchedSplitAttentionWorkspace,
        output: &mut DeviceBuffer<bf16>,
        batch_size: usize,
        window: Option<usize>,
        scale: f32,
        minimum_tokens: usize,
        maximum_tokens: usize,
    ) -> Result<()> {
        self.validate(
            query, block_tables, token_counts, block_counts, workspace, output, batch_size,
        )?;
        self.execute_partitions(
            stream, query, key_pages, value_pages, block_tables, token_counts, block_counts,
            workspace, output, batch_size, window, scale, minimum_tokens, maximum_tokens,
        )?;
        Ok(self.merge.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.query_heads)?, narrow(batch_size)?, 1),
                block: (128, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                &workspace.values,
                &workspace.maxima,
                &workspace.denominators,
                token_counts,
                output,
                narrow(batch_size)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.value_head_dim)?,
                narrow(window.unwrap_or(0))?,
                narrow(self.partition_tokens)?,
                narrow(self.max_partitions)?,
                narrow(minimum_tokens)?,
            ),
        )?)
    }

    #[must_use]
    pub(crate) const fn partition_tokens(&self) -> usize {
        self.partition_tokens
    }

    #[must_use]
    pub(crate) const fn max_partitions(&self) -> usize {
        self.max_partitions
    }

    #[must_use]
    pub(crate) const fn spec(&self) -> PagedAttentionSpec {
        self.spec
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        &self,
        query: &DeviceBuffer<bf16>,
        tables: &DeviceBuffer<u32>,
        tokens: &DeviceBuffer<u32>,
        blocks: &DeviceBuffer<u32>,
        workspace: &BatchedSplitAttentionWorkspace,
        output: &DeviceBuffer<bf16>,
        batch_size: usize,
    ) -> Result<()> {
        require(
            "batched split query",
            batch_size * self.spec.query_heads * self.spec.head_dim,
            query.len(),
        )?;
        require(
            "batched split output",
            batch_size * self.spec.query_heads * self.spec.value_head_dim,
            output.len(),
        )?;
        require("batched split tables", self.max_batch * self.spec.max_blocks, tables.len())?;
        require("batched split tokens", self.max_batch, tokens.len())?;
        require("batched split blocks", self.max_batch, blocks.len())?;
        let (values, statistics) = self.workspace_lengths();
        require("batched split values", values, workspace.values.len())?;
        require("batched split maxima", statistics, workspace.maxima.len())?;
        require("batched split denominators", statistics, workspace.denominators.len())?;
        Ok(())
    }
}

impl BatchedSplitAttentionWorkspace {
    #[must_use]
    pub const fn new(
        values: DeviceBuffer<f32>,
        maxima: DeviceBuffer<f32>,
        denominators: DeviceBuffer<f32>,
    ) -> Self {
        Self { values, maxima, denominators }
    }
}
