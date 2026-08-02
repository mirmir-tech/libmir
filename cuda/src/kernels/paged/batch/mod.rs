mod split;
mod store;

use mircuda::{
    Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::{PagedAttentionSpec, compile_options, validate_attention};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(BatchedAttentionKernel = "libmir_cuda_paged_attention_batch_bf16"(
    query: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>, block_tables: &DeviceBuffer<u32>,
    token_counts: &DeviceBuffer<u32>, block_counts: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, batch_size: u32, max_blocks: u32,
    block_size: u32, query_heads: u32, kv_heads: u32, head_dim: u32,
    value_head_dim: u32, window: u32, scale: f32, split_threshold: u32,
));

pub use split::{BatchedSplitAttentionWorkspace, BatchedSplitPagedAttention};

/// One decode-attention launch over independent paged sequences.
#[derive(Clone, Debug)]
pub struct BatchedPagedAttention {
    kernel: TypedKernel<BatchedAttentionKernel>,
    spec: PagedAttentionSpec,
    max_batch: usize,
}

impl BatchedPagedAttention {
    pub fn compile(
        compiler: &Compiler,
        spec: PagedAttentionSpec,
        max_batch: usize,
    ) -> Result<Self> {
        validate_attention(spec)?;
        if max_batch == 0 {
            return Err(Error::InvalidPagedKv("paged attention batch cannot be empty"));
        }
        let source = cuda_kernel_file!("../../../../kernels/paged_attention_batch_bf16.cu");
        let module = compiler.compile(source, &compile_options(spec.dtype)?)?;
        Ok(Self {
            kernel: module.kernel()?,
            spec,
            max_batch,
        })
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
        output: &mut DeviceBuffer<bf16>,
        batch_size: usize,
        window: Option<usize>,
        scale: f32,
        split_threshold: usize,
    ) -> Result<()> {
        let query_width = product(self.spec.query_heads, self.spec.head_dim)?;
        let output_width = product(self.spec.query_heads, self.spec.value_head_dim)?;
        require("batched attention query", product(batch_size, query_width)?, query.len())?;
        require("batched attention output", product(batch_size, output_width)?, output.len())?;
        require(
            "batched attention tables",
            product(self.max_batch, self.spec.max_blocks)?,
            block_tables.len(),
        )?;
        require("batched attention token counts", self.max_batch, token_counts.len())?;
        require("batched attention block counts", self.max_batch, block_counts.len())?;
        if batch_size == 0 || batch_size > self.max_batch || !scale.is_finite() {
            return Err(Error::InvalidPagedKv("invalid paged attention batch geometry"));
        }
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.query_heads)?, narrow(batch_size)?, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                query,
                key_pages,
                value_pages,
                block_tables,
                token_counts,
                block_counts,
                output,
                narrow(batch_size)?,
                narrow(self.spec.max_blocks)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.value_head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
                narrow(split_threshold)?,
            ),
        )?)
    }
}
