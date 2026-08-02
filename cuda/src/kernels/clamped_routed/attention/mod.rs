use mircuda::{
    CompileOptions, DeviceBuffer, FmhaBf16Plan, FmhaBf16Spec, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};
use runtime::kv::KvCacheDType;

use crate::{CudaBackend, Error, Result};
mod execution;
mod fmha;
mod split;
#[cfg(test)]
mod tests;
use execution::narrow;
pub use split::{ClampedRoutedBatchSplitDecode, ClampedRoutedSplitDecode};

cuda_export!(DecodeKernel = "libmir_cuda_clamped_routed_paged_attention_bf16"(
    query: &DeviceBuffer<bf16>, current_keys: &DeviceBuffer<bf16>,
    current_values: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>, block_table: &DeviceBuffer<u32>,
    sinks: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    token_count: u32, block_count: u32, block_size: u32, query_heads: u32,
    kv_heads: u32, head_dim: u32, window: u32, scale: f32,
));
cuda_export!(PrefillKernel = "libmir_cuda_clamped_routed_paged_prefill_attention_bf16"(
    query: &DeviceBuffer<bf16>, current_keys: &DeviceBuffer<bf16>,
    current_values: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
    value_pages: &DeviceBuffer<u8>, block_table: &DeviceBuffer<u32>,
    sinks: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    query_tokens: u32, start_position: u32, block_count: u32, block_size: u32,
    query_heads: u32, kv_heads: u32, head_dim: u32, window: u32, scale: f32,
));
cuda_export!(BatchPrefillKernel =
    "libmir_cuda_clamped_routed_paged_batch_prefill_attention_bf16"(
        query: &DeviceBuffer<bf16>, key_pages: &DeviceBuffer<u8>,
        current_keys: &DeviceBuffer<bf16>, current_values: &DeviceBuffer<bf16>,
        value_pages: &DeviceBuffer<u8>, block_tables: &DeviceBuffer<u32>,
        request_indices: &DeviceBuffer<u32>, positions: &DeviceBuffer<u32>,
        query_starts: &DeviceBuffer<u32>, block_counts: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, query_tokens: u32, max_blocks: u32,
        block_size: u32, query_heads: u32, kv_heads: u32, head_dim: u32,
        window: u32, scale: f32,
    )
);
cuda_export!(SinkScaleKernel = "libmir_cuda_clamped_routed_sink_scale_bf16"(
    output: &mut DeviceBuffer<bf16>, softmax_lse: &DeviceBuffer<bf16>,
    sinks: &DeviceBuffer<bf16>, query_tokens: u32, query_heads: u32,
    head_dim: u32,
));

#[derive(Debug)]
pub struct ClampedRoutedAttention {
    decode: TypedKernel<DecodeKernel>,
    prefill: TypedKernel<PrefillKernel>,
    batch_prefill: TypedKernel<BatchPrefillKernel>,
    sink_scale: TypedKernel<SinkScaleKernel>,
    fmha: Option<FmhaBf16Plan>,
    block_size: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl ClampedRoutedAttention {
    pub(crate) fn compile(
        backend: &CudaBackend,
        block_size: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        dtype: KvCacheDType,
        window: Option<usize>,
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
        let module = backend.compiler().compile(
            cuda_kernel_file!("../../../../kernels/clamped_routed_attention_bf16.cu"),
            &options,
        )?;
        let fmha = (window.is_none()
            && matches!(dtype, KvCacheDType::Auto | KvCacheDType::BFloat16)
            && matches!(head_dim, 64 | 128))
        .then(|| {
            FmhaBf16Plan::new(
                backend.context(),
                backend.stream(),
                FmhaBf16Spec::new(query_heads, kv_heads, head_dim, head_dim)?,
            )
        })
        .transpose()?;
        Ok(Self {
            decode: module.kernel()?,
            prefill: module.kernel()?,
            batch_prefill: module.kernel()?,
            sink_scale: module.kernel()?,
            fmha,
            block_size,
            query_heads,
            kv_heads,
            head_dim,
        })
    }
}
