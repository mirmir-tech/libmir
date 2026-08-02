use mircuda::{
    DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::{PagedAttentionSpec, compile_options, validate_attention};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

mod execution;
mod graph;
mod shape;

pub use graph::{
    MergeAttentionArguments, SplitAttentionArguments, SplitAttentionConfigs, SplitAttentionKernels,
    SplitAttentionNodes,
};
pub(super) use shape::tensor_queries;

cuda_export!(
    pub(crate) SplitAttentionKernel = "libmir_cuda_paged_attention_split_bf16"(
        query: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>, block_table: &DeviceBuffer<u32>,
        partial_values: &mut DeviceBuffer<f32>, partial_maxima: &mut DeviceBuffer<f32>,
        partial_denominators: &mut DeviceBuffer<f32>, token_count: u32, block_count: u32,
        block_size: u32, query_heads: u32, kv_heads: u32, head_dim: u32,
        value_head_dim: u32, window: u32, scale: f32, partition_tokens: u32,
        active_partitions: u32, max_partitions: u32, minimum_tokens: u32,
    )
);

cuda_export!(
    pub(crate) MergeAttentionKernel = "libmir_cuda_paged_attention_merge_bf16"(
        partial_values: &DeviceBuffer<f32>, partial_maxima: &DeviceBuffer<f32>,
        partial_denominators: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>,
        query_heads: u32, value_head_dim: u32, active_partitions: u32,
        max_partitions: u32, visible_tokens: u32, minimum_tokens: u32,
    )
);

#[derive(Clone, Debug)]
pub struct SplitPagedAttention {
    split: TypedKernel<SplitAttentionKernel>,
    merge: TypedKernel<MergeAttentionKernel>,
    spec: PagedAttentionSpec,
    grouped_queries: bool,
    tensor_queries: bool,
    partition_tokens: usize,
    max_partitions: usize,
}

impl SplitPagedAttention {
    pub fn compile(
        compiler: &mircuda::Compiler,
        spec: PagedAttentionSpec,
        partition_tokens: usize,
    ) -> Result<Self> {
        validate_attention(spec)?;
        if partition_tokens == 0 {
            return Err(Error::InvalidPagedKv("split-KV partition cannot be empty"));
        }
        let max_tokens = product(spec.max_blocks, spec.block_size)?;
        let max_partitions = max_tokens.div_ceil(partition_tokens);
        let query_group = spec.query_heads / spec.kv_heads;
        let grouped_queries = query_group > 1
            && query_group <= 8
            && spec.head_dim <= 128
            && spec.value_head_dim <= 128;
        let tensor_queries = tensor_queries(spec, query_group, grouped_queries);
        let source = cuda_kernel_file!("../../../../kernels/paged_attention_split_bf16.cu");
        let mut options = compile_options(spec.dtype)?;
        if grouped_queries {
            options.extra_options.push("-DLIBMIR_SPLIT_GQA=1".into());
            if tensor_queries {
                options.extra_options.push("-DLIBMIR_SPLIT_GQA_WMMA=1".into());
            }
            options.extra_options.push(format!("-DLIBMIR_QUERY_GROUP={query_group}"));
            options
                .extra_options
                .push(format!("-DLIBMIR_WARP_VALUE_ITEMS={}", spec.value_head_dim.div_ceil(32)));
        }
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            split: module.kernel()?,
            merge: module.kernel()?,
            spec,
            grouped_queries,
            tensor_queries,
            partition_tokens,
            max_partitions,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        block_table: &DeviceBuffer<u32>,
        workspace: &mut SplitAttentionWorkspace,
        output: &mut DeviceBuffer<bf16>,
        token_count: usize,
        block_count: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        let active = self.validate(
            query, block_table, workspace, output, token_count, block_count, window, scale,
        )?;
        let split_config = LaunchConfig {
            grid: (narrow(product(self.split_heads(), active)?)?, 1, 1),
            block: (self.threads(), 1, 1),
            shared_memory_bytes: 0,
        };
        self.split.launch(
            stream,
            split_config,
            (
                query,
                key_pages,
                value_pages,
                block_table,
                &mut workspace.values,
                &mut workspace.maxima,
                &mut workspace.denominators,
                narrow(window.map_or(token_count, |limit| token_count.min(limit)))?,
                narrow(block_count)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.value_head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
                narrow(self.partition_tokens)?,
                narrow(active)?,
                narrow(self.max_partitions)?,
                0,
            ),
        )?;
        let merge_config = LaunchConfig {
            grid: (narrow(self.spec.query_heads)?, 1, 1),
            block: (self.threads(), 1, 1),
            shared_memory_bytes: narrow(active * size_of::<f32>())?,
        };
        Ok(self.merge.launch(
            stream,
            merge_config,
            (
                &workspace.values,
                &workspace.maxima,
                &workspace.denominators,
                output,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.value_head_dim)?,
                narrow(active)?,
                narrow(self.max_partitions)?,
                narrow(token_count)?,
                0,
            ),
        )?)
    }

    #[must_use]
    pub const fn workspace_lengths(&self) -> (usize, usize) {
        (
            self.spec.query_heads * self.max_partitions * self.spec.value_head_dim,
            self.spec.query_heads * self.max_partitions,
        )
    }

    pub(crate) const fn spec(&self) -> PagedAttentionSpec {
        self.spec
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        &self,
        query: &DeviceBuffer<bf16>,
        block_table: &DeviceBuffer<u32>,
        workspace: &SplitAttentionWorkspace,
        output: &DeviceBuffer<bf16>,
        token_count: usize,
        block_count: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<usize> {
        require("split attention query", self.spec.query_heads * self.spec.head_dim, query.len())?;
        require(
            "split attention output",
            self.spec.query_heads * self.spec.value_head_dim,
            output.len(),
        )?;
        require("split attention block table", self.spec.max_blocks, block_table.len())?;
        let (value_len, statistic_len) = self.workspace_lengths();
        require("split attention values", value_len, workspace.values.len())?;
        require("split attention maxima", statistic_len, workspace.maxima.len())?;
        require("split attention denominators", statistic_len, workspace.denominators.len())?;
        let capacity = product(block_count, self.spec.block_size)?;
        if token_count == 0
            || block_count == 0
            || block_count > self.spec.max_blocks
            || token_count > capacity
            || !scale.is_finite()
        {
            return Err(Error::InvalidPagedKv("invalid split attention execution geometry"));
        }
        let visible = window.map_or(token_count, |limit| token_count.min(limit));
        Ok(visible.div_ceil(self.partition_tokens))
    }
}

#[derive(Debug)]
pub struct SplitAttentionWorkspace {
    pub(crate) values: DeviceBuffer<f32>,
    pub(crate) maxima: DeviceBuffer<f32>,
    pub(crate) denominators: DeviceBuffer<f32>,
}

impl SplitAttentionWorkspace {
    pub(crate) const fn new(
        values: DeviceBuffer<f32>,
        maxima: DeviceBuffer<f32>,
        denominators: DeviceBuffer<f32>,
    ) -> Self {
        Self { values, maxima, denominators }
    }
}
