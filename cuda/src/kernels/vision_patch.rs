use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::Result;

cuda_export!(
    PatchLayoutKernel = "libmir_cuda_vision_patch_cthw_to_thwc_bf16"(
        input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>, elements: u32,
        channels: u32, temporal: u32, patch_area: u32,
    )
);

#[derive(Clone, Debug)]
pub struct VisionPatchLayout {
    kernel: TypedKernel<PatchLayoutKernel>,
}

impl VisionPatchLayout {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../kernels/vision_patch_layout.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()? })
    }

    pub fn cthw_to_thwc(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
        geometry: [usize; 4],
    ) -> Result<()> {
        let [tokens, channels, temporal, patch_area] = geometry;
        let elements = product(product(product(tokens, channels)?, temporal)?, patch_area)?;
        require("vision patch layout input", elements, input.len())?;
        require("vision patch layout output", elements, output.len())?;
        let threads = 256_usize;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                output,
                narrow(elements)?,
                narrow(channels)?,
                narrow(temporal)?,
                narrow(patch_area)?,
            ),
        )?)
    }
}
