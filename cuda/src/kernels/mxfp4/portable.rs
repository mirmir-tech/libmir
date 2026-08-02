use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::MxFp4Spec;
use crate::{
    Result,
    kernels::geometry::{narrow, require},
};

cuda_export!(MxFp4Kernel = "libmir_cuda_mxfp4_bf16_linear"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>, scales: &DeviceBuffer<u8>,
    bias: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, tokens: u32,
    rows: u32, columns: u32, has_bias: u32,
));

#[derive(Clone, Debug)]
/// Portable direct-checkpoint OCP MXFP4 projection.
pub struct MxFp4Linear {
    kernel: TypedKernel<MxFp4Kernel>,
    spec: MxFp4Spec,
}

impl MxFp4Linear {
    pub fn compile(compiler: &Compiler, spec: MxFp4Spec) -> Result<Self> {
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/mxfp4_linear.cu"),
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
        scales: &DeviceBuffer<u8>,
        bias: Option<&DeviceBuffer<bf16>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("MXFP4 input", self.spec.input_elements()?, input.len())?;
        require("MXFP4 weight", self.spec.weight_elements()?, weight.len())?;
        require("MXFP4 scales", self.spec.scale_elements()?, scales.len())?;
        if let Some(bias) = bias {
            require("MXFP4 bias", self.spec.output_features, bias.len())?;
        }
        require("MXFP4 output", self.spec.output_elements()?, output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(self.spec.output_features.div_ceil(8))?,
                    narrow(self.spec.tokens)?,
                    1,
                ),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weight,
                scales,
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
