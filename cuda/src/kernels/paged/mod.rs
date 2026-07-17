use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};
use runtime::kv::KvCacheDType;

use super::geometry::product;
use crate::{Error, Result};

mod batch;
mod batch_store;
mod graph;
mod prefill;
mod split;
#[cfg(test)]
mod tests;

pub use batch::BatchedPagedAttention;
pub use prefill::PagedPrefillAttention;
pub use split::{
    MergeAttentionArguments, SplitAttentionArguments, SplitAttentionConfigs, SplitAttentionKernels,
    SplitAttentionNodes, SplitAttentionWorkspace, SplitPagedAttention,
};

cuda_export!(
    pub(crate) KvStoreKernel = "libmir_cuda_store_paged_kv_bf16"(
        keys: &DeviceBuffer<bf16>, values: &DeviceBuffer<bf16>,
        key_pages: &mut DeviceBuffer<u8>, value_pages: &mut DeviceBuffer<u8>,
        local_start: u32, token_count: u32, physical_block: u32, page_start: u32,
        block_size: u32, kv_heads: u32, key_head_dim: u32, value_head_dim: u32,
    )
);

cuda_export!(
    pub(crate) BatchKvStoreKernel = "libmir_cuda_store_paged_kv_batch_bf16"(
        keys: &DeviceBuffer<bf16>, values: &DeviceBuffer<bf16>,
        key_pages: &mut DeviceBuffer<u8>, value_pages: &mut DeviceBuffer<u8>,
        block_tables: &DeviceBuffer<u32>, token_counts: &DeviceBuffer<u32>,
        batch_size: u32, max_blocks: u32, block_size: u32, kv_heads: u32,
        key_head_dim: u32, value_head_dim: u32,
    )
);

cuda_export!(
    pub(crate) AttentionKernel = "libmir_cuda_paged_attention_bf16"(
        query: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>, block_table: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>, token_count: u32, block_count: u32,
        block_size: u32, query_heads: u32, kv_heads: u32, head_dim: u32,
        value_head_dim: u32, window: u32, scale: f32,
        split_threshold: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedKvSpec {
    pub block_size: usize,
    pub block_count: usize,
    pub kv_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub dtype: KvCacheDType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedAttentionSpec {
    pub block_size: usize,
    pub max_blocks: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_head_dim: usize,
    pub dtype: KvCacheDType,
}

#[derive(Clone, Debug)]
pub struct PagedKvStore {
    kernel: TypedKernel<KvStoreKernel>,
    batch_kernel: TypedKernel<BatchKvStoreKernel>,
    spec: PagedKvSpec,
}

#[derive(Clone, Debug)]
pub struct PagedAttention {
    kernel: TypedKernel<AttentionKernel>,
    spec: PagedAttentionSpec,
}

impl PagedKvStore {
    pub fn compile(compiler: &Compiler, spec: PagedKvSpec) -> Result<Self> {
        validate_kv(spec)?;
        let source = cuda_kernel_file!("../../../kernels/kv_cache_bf16.cu");
        let module = compiler.compile(source, &compile_options(spec.dtype)?)?;
        Ok(Self {
            kernel: module.kernel()?,
            batch_kernel: module.kernel()?,
            spec,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        keys: &DeviceBuffer<bf16>,
        values: &DeviceBuffer<bf16>,
        key_pages: &mut DeviceBuffer<u8>,
        value_pages: &mut DeviceBuffer<u8>,
        local_start: usize,
        token_count: usize,
        physical_block: usize,
        page_start: usize,
    ) -> Result<()> {
        let (config, arguments) = self.launch(
            keys, values, key_pages, value_pages, local_start, token_count, physical_block,
            page_start,
        )?;
        Ok(self.kernel.launch(stream, config, arguments)?)
    }

    pub fn key_bytes(&self) -> Result<usize> {
        page_bytes(self.spec, self.spec.key_head_dim)
    }

    pub fn value_bytes(&self) -> Result<usize> {
        page_bytes(self.spec, self.spec.value_head_dim)
    }

    pub(crate) fn kernel(&self) -> TypedKernel<KvStoreKernel> {
        self.kernel.clone()
    }
}

impl PagedAttention {
    pub fn compile(compiler: &Compiler, spec: PagedAttentionSpec) -> Result<Self> {
        validate_attention(spec)?;
        let source = cuda_kernel_file!("../../../kernels/paged_attention_bf16.cu");
        let module = compiler.compile(source, &compile_options(spec.dtype)?)?;
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
        token_count: usize,
        block_count: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        let (config, arguments) = self.launch(
            query, key_pages, value_pages, block_table, output, token_count, block_count, window,
            scale, 0,
        )?;
        Ok(self.kernel.launch(stream, config, arguments)?)
    }

    pub(crate) fn kernel(&self) -> TypedKernel<AttentionKernel> {
        self.kernel.clone()
    }
}

fn page_bytes(spec: PagedKvSpec, head_dim: usize) -> Result<usize> {
    let elements =
        product(product(product(spec.block_count, spec.block_size)?, spec.kv_heads)?, head_dim)?;
    Ok(elements * usize::from(element_bytes(spec.dtype)?))
}

fn compile_options(dtype: KvCacheDType) -> Result<CompileOptions> {
    match dtype {
        KvCacheDType::Auto | KvCacheDType::BFloat16 => Ok(CompileOptions::default()),
        KvCacheDType::Fp8 | KvCacheDType::Fp8E4M3 => Ok(CompileOptions {
            extra_options: vec!["--define-macro=LIBMIR_KV_FP8=1".into()],
            ..CompileOptions::default()
        }),
        _ => Err(Error::InvalidPagedKv("unsupported CUDA KV cache dtype")),
    }
}

fn element_bytes(dtype: KvCacheDType) -> Result<u8> {
    match dtype {
        KvCacheDType::Auto | KvCacheDType::BFloat16 => Ok(2),
        KvCacheDType::Fp8 | KvCacheDType::Fp8E4M3 => Ok(1),
        _ => Err(Error::InvalidPagedKv("unsupported CUDA KV cache dtype")),
    }
}

fn validate_kv(spec: PagedKvSpec) -> Result<()> {
    let _ = element_bytes(spec.dtype)?;
    if spec.block_size == 0
        || spec.block_count == 0
        || spec.kv_heads == 0
        || spec.key_head_dim == 0
        || spec.value_head_dim == 0
    {
        Err(Error::InvalidPagedKv("invalid paged KV geometry"))
    } else {
        Ok(())
    }
}

fn validate_attention(spec: PagedAttentionSpec) -> Result<()> {
    let _ = element_bytes(spec.dtype)?;
    if spec.block_size == 0
        || spec.max_blocks == 0
        || spec.query_heads == 0
        || spec.kv_heads == 0
        || !spec.query_heads.is_multiple_of(spec.kv_heads)
        || spec.head_dim == 0
        || spec.head_dim > 512
        || spec.value_head_dim == 0
        || spec.value_head_dim > 512
    {
        Err(Error::InvalidPagedKv("invalid paged attention geometry"))
    } else {
        Ok(())
    }
}
