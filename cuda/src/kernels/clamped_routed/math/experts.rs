use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16, cuda_export};

use super::{ClampedRoutedKernels, narrow};
use crate::Result;

cuda_export!(pub(super) GateUpKernel = "libmir_cuda_clamped_routed_mxfp4_gate_up_bf16"(
    input: &DeviceBuffer<bf16>, blocks: &DeviceBuffer<u8>, scales: &DeviceBuffer<u8>,
    bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, tokens: u32, top_k: u32, hidden: u32,
    intermediate: u32, limit: f32,
));
cuda_export!(pub(super) DownKernel = "libmir_cuda_clamped_routed_mxfp4_down_bf16"(
    input: &DeviceBuffer<bf16>, blocks: &DeviceBuffer<u8>, scales: &DeviceBuffer<u8>,
    bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, tokens: u32,
    top_k: u32, hidden: u32, intermediate: u32,
));
cuda_export!(pub(super) GateUpMlxKernel = "libmir_cuda_clamped_routed_mlx_mxfp4_gate_up_bf16"(
    input: &DeviceBuffer<bf16>, gate_blocks: &DeviceBuffer<u32>,
    gate_scales: &DeviceBuffer<u8>, gate_bias: &DeviceBuffer<bf16>,
    up_blocks: &DeviceBuffer<u32>, up_scales: &DeviceBuffer<u8>,
    up_bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, tokens: u32, top_k: u32, hidden: u32,
    intermediate: u32, limit: f32,
));
cuda_export!(pub(super) DownMlxKernel = "libmir_cuda_clamped_routed_mlx_mxfp4_down_bf16"(
    input: &DeviceBuffer<bf16>, blocks: &DeviceBuffer<u32>, scales: &DeviceBuffer<u8>,
    bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, tokens: u32,
    top_k: u32, hidden: u32, intermediate: u32,
));

impl ClampedRoutedKernels {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gate_up_native(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        blocks: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        bias: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.gate_up.launch(
            stream,
            self.gate_launch()?,
            (
                input,
                blocks,
                scales,
                bias,
                selected,
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.top_k)?,
                narrow(self.spec.hidden)?,
                narrow(self.spec.intermediate)?,
                self.spec.swiglu_limit,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn down_native(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        blocks: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        bias: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.down.launch(
            stream,
            self.down_launch()?,
            (
                input,
                blocks,
                scales,
                bias,
                selected,
                routing,
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.top_k)?,
                narrow(self.spec.hidden)?,
                narrow(self.spec.intermediate)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gate_up_mlx(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        gate_blocks: &DeviceBuffer<u32>,
        gate_scales: &DeviceBuffer<u8>,
        gate_bias: &DeviceBuffer<bf16>,
        up_blocks: &DeviceBuffer<u32>,
        up_scales: &DeviceBuffer<u8>,
        up_bias: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.gate_up_mlx.launch(
            stream,
            self.gate_launch()?,
            (
                input,
                gate_blocks,
                gate_scales,
                gate_bias,
                up_blocks,
                up_scales,
                up_bias,
                selected,
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.top_k)?,
                narrow(self.spec.hidden)?,
                narrow(self.spec.intermediate)?,
                self.spec.swiglu_limit,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn down_mlx(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        blocks: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<u8>,
        bias: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.down_mlx.launch(
            stream,
            self.down_launch()?,
            (
                input,
                blocks,
                scales,
                bias,
                selected,
                routing,
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.top_k)?,
                narrow(self.spec.hidden)?,
                narrow(self.spec.intermediate)?,
            ),
        )?)
    }

    fn gate_launch(&self) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            grid: (narrow(self.spec.intermediate * self.spec.tokens * self.spec.top_k)?, 1, 1),
            block: (32, 1, 1),
            shared_memory_bytes: 0,
        })
    }

    fn down_launch(&self) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            grid: (narrow(self.spec.hidden * self.spec.tokens)?, 1, 1),
            block: (32, 1, 1),
            shared_memory_bytes: 0,
        })
    }
}
