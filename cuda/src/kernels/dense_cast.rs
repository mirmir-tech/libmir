use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file, f16,
};

use crate::{Error, Result};

cuda_export!(
    F16ToBf16Kernel = "libmir_cuda_dense_f16_to_bf16"(
        input: &DeviceBuffer<f16>, output: &mut DeviceBuffer<bf16>, elements: u32,
    )
);
cuda_export!(
    F32ToBf16Kernel = "libmir_cuda_dense_f32_to_bf16"(
        input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>, elements: u32,
    )
);

#[derive(Debug)]
pub struct DenseCast {
    f16_to_bf16: TypedKernel<F16ToBf16Kernel>,
    f32_to_bf16: TypedKernel<F32ToBf16Kernel>,
}

impl DenseCast {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../kernels/dense_cast.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            f16_to_bf16: module.kernel()?,
            f32_to_bf16: module.kernel()?,
        })
    }

    pub fn f16_to_bf16(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        validate(input.len(), output.len())?;
        Ok(self.f16_to_bf16.launch(
            stream,
            launch(input.len())?,
            (input, output, u32::try_from(input.len())?),
        )?)
    }

    pub fn f32_to_bf16(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        validate(input.len(), output.len())?;
        Ok(self.f32_to_bf16.launch(
            stream,
            launch(input.len())?,
            (input, output, u32::try_from(input.len())?),
        )?)
    }
}

fn validate(input: usize, output: usize) -> Result<()> {
    if input == 0 || input != output {
        return Err(Error::InvalidTensorConversion { input, output });
    }
    Ok(())
}

fn launch(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig::for_elements(elements, 256)?)
}
