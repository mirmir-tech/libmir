use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{DirectFp8Activation, DirectFp8Format, DirectFp8Scale, DirectFp8Spec};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, require},
};

cuda_export!(E5M2TensorCoreKernel = "libmir_cuda_e5m2_bf16_tensor_core_linear"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>, bias: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>, tokens: u32, rows: u32, columns: u32, has_bias: u32,
));

#[derive(Clone, Debug)]
/// SM120 weight-only E5M2 projection with tiled BF16 Tensor Core execution.
pub struct DirectE5M2WeightOnlyTensorCoreLinear {
    kernel: TypedKernel<E5M2TensorCoreKernel>,
    spec: DirectFp8Spec,
}

impl DirectE5M2WeightOnlyTensorCoreLinear {
    pub fn compile(compiler: &Compiler, spec: DirectFp8Spec) -> Result<Self> {
        if spec.format != DirectFp8Format::E5M2
            || spec.activation != DirectFp8Activation::Bf16
            || spec.scale != DirectFp8Scale::Tensor
            || spec.inverse_scale
            || !spec.input_features.is_multiple_of(16)
            || !spec.output_features.is_multiple_of(16)
        {
            return Err(Error::InvalidExecutionPlan(
                "E5M2 weight-only Tensor Core contract is unavailable",
            ));
        }
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/direct_fp8_e5m2_tensor_core.cu"),
            &CompileOptions {
                fast_math: false,
                ..CompileOptions::default()
            },
        )?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        bias: Option<&DeviceBuffer<bf16>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("E5M2 Tensor Core input", self.spec.input_elements()?, input.len())?;
        require("E5M2 Tensor Core weight", self.spec.weight_elements()?, weight.len())?;
        require("E5M2 Tensor Core output", self.spec.output_elements()?, output.len())?;
        if let Some(bias) = bias {
            require("E5M2 Tensor Core bias", self.spec.output_features, bias.len())?;
        }
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(self.spec.output_features.div_ceil(128))?,
                    narrow(self.spec.tokens.div_ceil(16))?,
                    1,
                ),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weight,
                bias.unwrap_or(input),
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.output_features)?,
                narrow(self.spec.input_features)?,
                u32::from(bias.is_some()),
            ),
        )?)
    }
}
