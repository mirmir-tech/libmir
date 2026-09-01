use mircuda::{
    CompileOptions, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{CudaBackend, Result};

cuda_export!(StageWindowedPrefillKernel = "libmir_cuda_stage_windowed_prefill_kv_bf16"(
    current_keys: &DeviceBuffer<bf16>, current_values: &DeviceBuffer<bf16>,
    ring_keys: &DeviceBuffer<u8>, ring_values: &DeviceBuffer<u8>,
    staged_keys: &mut DeviceBuffer<u8>, staged_values: &mut DeviceBuffer<u8>,
    ring_tables: &DeviceBuffer<u32>, query_starts: &DeviceBuffer<u32>,
    source_starts: &DeviceBuffer<u32>, history_tokens: &DeviceBuffer<u32>,
    context_tokens: &DeviceBuffer<u32>, active_rows: u32, max_context_tokens: u32,
    ring_max_blocks: u32, staged_blocks_per_row: u32, block_size: u32,
    kv_heads: u32, head_dim: u32,
));

#[derive(Debug)]
pub struct WindowedPrefillStage {
    kernel: TypedKernel<StageWindowedPrefillKernel>,
}

pub struct WindowedPrefillStageArgs<'a> {
    pub current_keys: &'a DeviceBuffer<bf16>,
    pub current_values: &'a DeviceBuffer<bf16>,
    pub ring_keys: &'a DeviceBuffer<u8>,
    pub ring_values: &'a DeviceBuffer<u8>,
    pub staged_keys: &'a mut DeviceBuffer<u8>,
    pub staged_values: &'a mut DeviceBuffer<u8>,
    pub ring_tables: &'a DeviceBuffer<u32>,
    pub query_starts: &'a DeviceBuffer<u32>,
    pub source_starts: &'a DeviceBuffer<u32>,
    pub history_tokens: &'a DeviceBuffer<u32>,
    pub context_tokens: &'a DeviceBuffer<u32>,
    pub active_rows: usize,
    pub max_context_tokens: usize,
    pub ring_max_blocks: usize,
    pub staged_blocks_per_row: usize,
    pub block_size: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl WindowedPrefillStage {
    pub(crate) fn compile(backend: &CudaBackend) -> Result<Self> {
        let module = backend.compiler().compile(
            cuda_kernel_file!("../../kernels/windowed_prefill_staging_bf16.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self { kernel: module.kernel()? })
    }

    pub(crate) fn execute(
        &self,
        stream: &Stream,
        args: &mut WindowedPrefillStageArgs<'_>,
    ) -> Result<()> {
        let width = args
            .kv_heads
            .checked_mul(args.head_dim)
            .ok_or(crate::Error::InvalidPagedKv("windowed prefill staging width overflow"))?;
        let elements = args
            .active_rows
            .checked_mul(args.max_context_tokens)
            .and_then(|value| value.checked_mul(width))
            .ok_or(crate::Error::InvalidPagedKv("windowed prefill staging launch overflow"))?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig::for_elements(elements, 256)?,
            (
                args.current_keys,
                args.current_values,
                args.ring_keys,
                args.ring_values,
                args.staged_keys,
                args.staged_values,
                args.ring_tables,
                args.query_starts,
                args.source_starts,
                args.history_tokens,
                args.context_tokens,
                u32::try_from(args.active_rows)?,
                u32::try_from(args.max_context_tokens)?,
                u32::try_from(args.ring_max_blocks)?,
                u32::try_from(args.staged_blocks_per_row)?,
                u32::try_from(args.block_size)?,
                u32::try_from(args.kv_heads)?,
                u32::try_from(args.head_dim)?,
            ),
        )?)
    }
}
