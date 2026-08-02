use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::NvFp4Spec;
use crate::{Error, Result, kernels::geometry::require};

cuda_export!(
    NvFp4WeightOnlyKernel = "libmir_cuda_nvfp4_weight_only_bf16"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>,
        block_scales: &DeviceBuffer<u8>, global_scale: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>, input_features: u32,
        output_features: u32, tokens: u32,
    )
);

#[derive(Clone, Debug)]
pub struct NvFp4WeightOnly {
    kernel: TypedKernel<NvFp4WeightOnlyKernel>,
    spec: NvFp4Spec,
    tokens: usize,
}

pub struct NvFp4WeightOnlyLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub weight: &'a DeviceBuffer<u8>,
    pub block_scales: &'a DeviceBuffer<u8>,
    pub global_scale: &'a DeviceBuffer<f32>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl NvFp4WeightOnly {
    pub fn compile(compiler: &Compiler, spec: NvFp4Spec, tokens: usize) -> Result<Self> {
        if tokens == 0 {
            return Err(Error::InvalidNvFp4("weight-only token count is empty"));
        }
        let source = cuda_kernel_file!("../../../kernels/nvfp4_weight_only_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec, tokens })
    }

    pub fn execute(&self, stream: &Stream, launch: &mut NvFp4WeightOnlyLaunch<'_>) -> Result<()> {
        let elements = self.spec.elements()?;
        require(
            "NVFP4 weight-only input",
            self.tokens * self.spec.input_features,
            launch.input.len(),
        )?;
        require("NVFP4 weight-only weight", elements / 2, launch.weight.len())?;
        require("NVFP4 weight-only scales", elements / 16, launch.block_scales.len())?;
        require("NVFP4 weight-only global", 1, launch.global_scale.len())?;
        require(
            "NVFP4 weight-only output",
            self.tokens * self.spec.output_features,
            launch.output.len(),
        )?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (
                    u32::try_from(self.spec.output_features.div_ceil(8))?,
                    u32::try_from(self.tokens)?,
                    1,
                ),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                launch.input,
                launch.weight,
                launch.block_scales,
                launch.global_scale,
                launch.output,
                u32::try_from(self.spec.input_features)?,
                u32::try_from(self.spec.output_features)?,
                u32::try_from(self.tokens)?,
            ),
        )?)
    }
}
