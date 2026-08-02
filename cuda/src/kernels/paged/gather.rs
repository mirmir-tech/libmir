use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, cuda_export,
    cuda_kernel_file,
};

use super::{PagedAttentionSpec, validate_attention};
use crate::{
    Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(PagedKvGatherKernel = "libmir_cuda_gather_paged_kv_bf16"(
    key_pages: &DeviceBuffer<u8>, value_pages: &DeviceBuffer<u8>,
    block_table: &DeviceBuffer<u32>, keys: &mut DeviceBuffer<u8>,
    values: &mut DeviceBuffer<u8>, context_tokens: u32, block_count: u32,
    block_size: u32, key_width: u32, value_width: u32,
));

cuda_export!(BatchedPagedKvGatherKernel = "libmir_cuda_gather_paged_kv_batch_bf16"(
    key_pages: &DeviceBuffer<u8>, value_pages: &DeviceBuffer<u8>,
    block_tables: &DeviceBuffer<u32>, context_starts: &DeviceBuffer<u32>,
    keys: &mut DeviceBuffer<mircuda::bf16>, values: &mut DeviceBuffer<mircuda::bf16>,
    batch_size: u32, max_blocks: u32, block_size: u32,
    key_width: u32, value_width: u32,
));

/// Gathers logical BF16 K/V rows from physical pages into contiguous buffers.
#[derive(Clone, Debug)]
pub struct PagedKvGather {
    kernel: TypedKernel<PagedKvGatherKernel>,
    spec: PagedAttentionSpec,
}

#[derive(Clone, Debug)]
pub struct BatchedPagedKvGather {
    kernel: TypedKernel<BatchedPagedKvGatherKernel>,
    spec: PagedAttentionSpec,
}

impl PagedKvGather {
    pub fn compile(compiler: &Compiler, spec: PagedAttentionSpec) -> Result<Self> {
        validate_attention(spec)?;
        let source = cuda_kernel_file!("../../../kernels/paged_gather_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        block_table: &DeviceBuffer<u32>,
        keys: &mut DeviceBuffer<u8>,
        values: &mut DeviceBuffer<u8>,
        context_tokens: usize,
        block_count: usize,
    ) -> Result<()> {
        let key_width = product(self.spec.kv_heads, self.spec.head_dim)?;
        let value_width = product(self.spec.kv_heads, self.spec.value_head_dim)?;
        require(
            "gathered prefill keys",
            product(product(context_tokens, key_width)?, size_of::<u16>())?,
            keys.len(),
        )?;
        require(
            "gathered prefill values",
            product(product(context_tokens, value_width)?, size_of::<u16>())?,
            values.len(),
        )?;
        require("gathered prefill table", self.spec.max_blocks, block_table.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(context_tokens)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                key_pages,
                value_pages,
                block_table,
                keys,
                values,
                narrow(context_tokens)?,
                narrow(block_count)?,
                narrow(self.spec.block_size)?,
                narrow(key_width)?,
                narrow(value_width)?,
            ),
        )?)
    }
}

impl BatchedPagedKvGather {
    pub fn compile(compiler: &Compiler, spec: PagedAttentionSpec) -> Result<Self> {
        validate_attention(spec)?;
        let source = cuda_kernel_file!("../../../kernels/paged_gather_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        block_tables: &DeviceBuffer<u32>,
        context_starts: &DeviceBuffer<u32>,
        keys: &mut DeviceBuffer<mircuda::bf16>,
        values: &mut DeviceBuffer<mircuda::bf16>,
        batch_size: usize,
        total_context_tokens: usize,
        max_context_tokens: usize,
    ) -> Result<()> {
        let key_width = product(self.spec.kv_heads, self.spec.head_dim)?;
        let value_width = product(self.spec.kv_heads, self.spec.value_head_dim)?;
        require(
            "batched gathered prefill keys",
            product(total_context_tokens, key_width)?,
            keys.len(),
        )?;
        require(
            "batched gathered prefill values",
            product(total_context_tokens, value_width)?,
            values.len(),
        )?;
        require(
            "batched gathered prefill tables",
            product(batch_size, self.spec.max_blocks)?,
            block_tables.len(),
        )?;
        require("batched gathered prefill starts", batch_size + 1, context_starts.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(max_context_tokens)?, narrow(batch_size)?, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                key_pages,
                value_pages,
                block_tables,
                context_starts,
                keys,
                values,
                narrow(batch_size)?,
                narrow(self.spec.max_blocks)?,
                narrow(self.spec.block_size)?,
                narrow(key_width)?,
                narrow(value_width)?,
            ),
        )?)
    }
}
