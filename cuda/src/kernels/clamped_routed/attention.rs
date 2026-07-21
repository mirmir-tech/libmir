use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};
use runtime::kv::KvCacheDType;

use crate::{Error, Result};

cuda_export!(DecodeKernel = "libmir_cuda_clamped_routed_paged_attention_bf16"(
    query: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>, block_table: &DeviceBuffer<u32>,
    sinks: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    token_count: u32, block_count: u32, block_size: u32, query_heads: u32,
    kv_heads: u32, head_dim: u32, window: u32, scale: f32,
));
cuda_export!(PrefillKernel = "libmir_cuda_clamped_routed_paged_prefill_attention_bf16"(
    query: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>, block_table: &DeviceBuffer<u32>,
    sinks: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    query_tokens: u32, start_position: u32, block_count: u32, block_size: u32,
    query_heads: u32, kv_heads: u32, head_dim: u32, window: u32, scale: f32,
));

#[derive(Clone, Debug)]
pub struct ClampedRoutedAttention {
    decode: TypedKernel<DecodeKernel>,
    prefill: TypedKernel<PrefillKernel>,
    block_size: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl ClampedRoutedAttention {
    pub(crate) fn compile(
        compiler: &Compiler,
        block_size: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        dtype: KvCacheDType,
    ) -> Result<Self> {
        if block_size == 0
            || query_heads == 0
            || kv_heads == 0
            || !query_heads.is_multiple_of(kv_heads)
            || head_dim == 0
            || head_dim > 256
        {
            return Err(Error::InvalidPagedKv("invalid clamped-routed attention geometry"));
        }
        let options = match dtype {
            KvCacheDType::Auto | KvCacheDType::BFloat16 => CompileOptions::default(),
            KvCacheDType::Fp8 | KvCacheDType::Fp8E4M3 => CompileOptions {
                extra_options: vec!["--define-macro=LIBMIR_KV_FP8=1".into()],
                ..CompileOptions::default()
            },
            _ => return Err(Error::InvalidPagedKv("unsupported clamped-routed KV dtype")),
        };
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/clamped_routed_attention_bf16.cu"),
            &options,
        )?;
        Ok(Self {
            decode: module.kernel()?,
            prefill: module.kernel()?,
            block_size,
            query_heads,
            kv_heads,
            head_dim,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute(
        &self,
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
    ) -> Result<()> {
        Ok(self.decode.launch(
            stream,
            Self::launch(self.query_heads)?,
            (
                query,
                key_pages,
                value_pages,
                table,
                sinks,
                output,
                narrow(tokens)?,
                narrow(blocks)?,
                narrow(self.block_size)?,
                narrow(self.query_heads)?,
                narrow(self.kv_heads)?,
                narrow(self.head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_prefill(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        table: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        query_tokens: usize,
        start: usize,
        blocks: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        Ok(self.prefill.launch(
            stream,
            Self::launch(query_tokens * self.query_heads)?,
            (
                query,
                key_pages,
                value_pages,
                table,
                sinks,
                output,
                narrow(query_tokens)?,
                narrow(start)?,
                narrow(blocks)?,
                narrow(self.block_size)?,
                narrow(self.query_heads)?,
                narrow(self.kv_heads)?,
                narrow(self.head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
            ),
        )?)
    }

    fn launch(blocks: usize) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            grid: (narrow(blocks)?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        })
    }
}

fn narrow(value: usize) -> Result<u32> {
    Ok(u32::try_from(value)?)
}
