use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16, cuda_export};

use super::{ClampedRoutedKernels, linear_launch, narrow};
use crate::Result;

cuda_export!(pub(super) DownRoutesKernel = "libmir_cuda_clamped_routed_mxfp4_down_routes_bf16"(
    input: &DeviceBuffer<bf16>, blocks: &DeviceBuffer<u8>, scales: &DeviceBuffer<u8>,
    bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>, partial: &mut DeviceBuffer<f32>, tokens: u32,
    top_k: u32, hidden: u32, intermediate: u32,
));
cuda_export!(pub(super) DownMlxRoutesKernel = "libmir_cuda_clamped_routed_mlx_mxfp4_down_routes_bf16"(
    input: &DeviceBuffer<bf16>, blocks: &DeviceBuffer<u32>, scales: &DeviceBuffer<u8>,
    bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>, partial: &mut DeviceBuffer<f32>, tokens: u32,
    top_k: u32, hidden: u32, intermediate: u32,
));
cuda_export!(pub(super) ReduceRoutesKernel = "libmir_cuda_clamped_routed_reduce_routes_bf16"(
    partial: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>, tokens: u32,
    top_k: u32, hidden: u32,
));

impl ClampedRoutedKernels {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn down_routes_native(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        blocks: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        bias: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        partial: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.down_routes.launch(
            stream,
            self.route_launch()?,
            (
                input,
                blocks,
                scales,
                bias,
                selected,
                routing,
                &mut *partial,
                narrow(self.spec.tokens)?,
                narrow(self.spec.top_k)?,
                narrow(self.spec.hidden)?,
                narrow(self.spec.intermediate)?,
            ),
        )?;
        self.reduce_routes(stream, partial, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn down_routes_mlx(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        blocks: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<u8>,
        bias: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        partial: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.down_mlx_routes.launch(
            stream,
            self.route_launch()?,
            (
                input,
                blocks,
                scales,
                bias,
                selected,
                routing,
                &mut *partial,
                narrow(self.spec.tokens)?,
                narrow(self.spec.top_k)?,
                narrow(self.spec.hidden)?,
                narrow(self.spec.intermediate)?,
            ),
        )?;
        self.reduce_routes(stream, partial, output)
    }

    fn reduce_routes(
        &self,
        stream: &Stream,
        partial: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.reduce_routes.launch(
            stream,
            linear_launch(self.spec.tokens * self.spec.hidden)?,
            (
                partial,
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.top_k)?,
                narrow(self.spec.hidden)?,
            ),
        )?)
    }

    fn route_launch(&self) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            grid: (narrow(self.spec.hidden * self.spec.tokens * self.spec.top_k)?, 1, 1),
            block: (32, 1, 1),
            shared_memory_bytes: 0,
        })
    }
}
