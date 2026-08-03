use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::NvFp4Spec;
use crate::{Error, Result, kernels::geometry::require};

cuda_export!(TensorCoreKernel = "libmir_cuda_nvfp4_weight_only_tensor_core_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>,
    block_scales: &DeviceBuffer<u8>, global_scale: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>, input_features: u32,
    output_features: u32, tokens: u32,
));

#[derive(Clone, Debug)]
pub struct NvFp4WeightOnlyTensorCore {
    kernel: TypedKernel<TensorCoreKernel>,
    spec: NvFp4Spec,
    tokens: usize,
}

impl NvFp4WeightOnlyTensorCore {
    pub fn compile(compiler: &Compiler, spec: NvFp4Spec, tokens: usize) -> Result<Self> {
        if tokens == 0 || !spec.input_features.is_multiple_of(16) {
            return Err(Error::InvalidNvFp4("invalid weight-only Tensor Core geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/nvfp4_weight_only_tensor_core.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self { kernel: module.kernel()?, spec, tokens })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        launch: &mut super::NvFp4WeightOnlyLaunch<'_>,
    ) -> Result<()> {
        let elements = self.spec.elements()?;
        require(
            "NVFP4 Tensor Core input",
            self.tokens * self.spec.input_features,
            launch.input.len(),
        )?;
        require("NVFP4 Tensor Core weight", elements / 2, launch.weight.len())?;
        require("NVFP4 Tensor Core scales", elements / 16, launch.block_scales.len())?;
        require("NVFP4 Tensor Core global", 1, launch.global_scale.len())?;
        require(
            "NVFP4 Tensor Core output",
            self.tokens * self.spec.output_features,
            launch.output.len(),
        )?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (
                    u32::try_from(self.spec.output_features.div_ceil(128))?,
                    u32::try_from(self.tokens.div_ceil(16))?,
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
                &mut *launch.output,
                u32::try_from(self.spec.input_features)?,
                u32::try_from(self.spec.output_features)?,
                u32::try_from(self.tokens)?,
            ),
        )?)
    }
}
