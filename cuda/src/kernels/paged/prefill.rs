use mircuda::{
    Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::{PagedAttentionSpec, validate_attention};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(PrefillAttentionKernel = "libmir_cuda_paged_prefill_attention_bf16"(
    query: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>, block_table: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, query_tokens: u32, start_position: u32,
    block_count: u32, block_size: u32, query_heads: u32, kv_heads: u32,
    head_dim: u32, value_head_dim: u32, window: u32, scale: f32,
    image_start: u32, image_end: u32,
));

#[derive(Clone, Debug)]
pub struct PagedPrefillAttention {
    kernel: TypedKernel<PrefillAttentionKernel>,
    spec: PagedAttentionSpec,
}

impl PagedPrefillAttention {
    pub fn compile(compiler: &Compiler, spec: PagedAttentionSpec) -> Result<Self> {
        validate_attention(spec)?;
        let source = cuda_kernel_file!("../../../kernels/paged_prefill_attention_bf16.cu");
        let module = compiler.compile(source, &super::compile_options(spec.dtype)?)?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        block_table: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
        query_tokens: usize,
        start_position: usize,
        block_count: usize,
        window: Option<usize>,
        scale: f32,
        image: Option<(usize, usize)>,
    ) -> Result<()> {
        let query_width = product(self.spec.query_heads, self.spec.head_dim)?;
        let output_width = product(self.spec.query_heads, self.spec.value_head_dim)?;
        require("prefill attention query", product(query_tokens, query_width)?, query.len())?;
        require("prefill attention output", product(query_tokens, output_width)?, output.len())?;
        require("prefill attention table", self.spec.max_blocks, block_table.len())?;
        let context = start_position
            .checked_add(query_tokens)
            .ok_or(Error::InvalidPagedKv("prefill attention context overflow"))?;
        let capacity = product(block_count, self.spec.block_size)?;
        if query_tokens == 0
            || block_count == 0
            || block_count > self.spec.max_blocks
            || context > capacity
            || !scale.is_finite()
            || image.is_some_and(|(start, end)| start >= end || end > context)
        {
            return Err(Error::InvalidPagedKv("invalid prefill attention geometry"));
        }
        let blocks = product(query_tokens, self.spec.query_heads)?;
        let (image_start, image_end) = image.unwrap_or((0, 0));
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
                block_table,
                output,
                narrow(query_tokens)?,
                narrow(start_position)?,
                narrow(block_count)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.value_head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
                narrow(image_start)?,
                narrow(image_end)?,
            ),
        )?)
    }
}
