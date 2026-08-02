use mircuda::{
    Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::{PagedAttentionSpec, validate_attention};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(BatchedPrefillAttentionKernel =
    "libmir_cuda_paged_prefill_attention_batch_bf16"(
        query: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>, block_tables: &DeviceBuffer<u32>,
        block_counts: &DeviceBuffer<u32>, request_indices: &DeviceBuffer<u32>,
        positions: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>,
        query_tokens: u32, batch_size: u32, max_blocks: u32, block_size: u32,
        query_heads: u32, kv_heads: u32, head_dim: u32, value_head_dim: u32,
        window: u32, scale: f32,
    )
);

#[derive(Clone, Debug)]
pub struct BatchedPagedPrefillAttention {
    kernel: TypedKernel<BatchedPrefillAttentionKernel>,
    spec: PagedAttentionSpec,
    max_batch: usize,
}

impl BatchedPagedPrefillAttention {
    pub fn compile(
        compiler: &Compiler,
        spec: PagedAttentionSpec,
        max_batch: usize,
    ) -> Result<Self> {
        validate_attention(spec)?;
        if max_batch == 0 {
            return Err(Error::InvalidPagedKv("paged prefill attention batch is empty"));
        }
        let source = cuda_kernel_file!("../../../kernels/paged_prefill_attention_batch_bf16.cu");
        let module = compiler.compile(source, &super::compile_options(spec.dtype)?)?;
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
        block_counts: &DeviceBuffer<u32>,
        request_indices: &DeviceBuffer<u32>,
        positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
        query_tokens: usize,
        batch_size: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        self.validate(
            query, block_tables, block_counts, request_indices, positions, output, query_tokens,
            batch_size, scale,
        )?;
        let blocks = product(query_tokens, self.spec.query_heads)?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(blocks)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                query,
                key_pages,
                value_pages,
                block_tables,
                block_counts,
                request_indices,
                positions,
                output,
                narrow(query_tokens)?,
                narrow(batch_size)?,
                narrow(self.spec.max_blocks)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.value_head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        &self,
        query: &DeviceBuffer<bf16>,
        tables: &DeviceBuffer<u32>,
        blocks: &DeviceBuffer<u32>,
        requests: &DeviceBuffer<u32>,
        positions: &DeviceBuffer<u32>,
        output: &DeviceBuffer<bf16>,
        tokens: usize,
        batch: usize,
        scale: f32,
    ) -> Result<()> {
        let query_width = product(self.spec.query_heads, self.spec.head_dim)?;
        let output_width = product(self.spec.query_heads, self.spec.value_head_dim)?;
        require("batched prefill query", product(tokens, query_width)?, query.len())?;
        require("batched prefill output", product(tokens, output_width)?, output.len())?;
        require("batched prefill tables", product(batch, self.spec.max_blocks)?, tables.len())?;
        require("batched prefill block counts", batch, blocks.len())?;
        require("batched prefill requests", tokens, requests.len())?;
        require("batched prefill positions", tokens, positions.len())?;
        if tokens == 0 || batch == 0 || batch > self.max_batch || !scale.is_finite() {
            return Err(Error::InvalidPagedKv("invalid batched prefill attention geometry"));
        }
        Ok(())
    }
}
