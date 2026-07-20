use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{Error, Result};

cuda_export!(
    NormalizeKernel = "libmir_cuda_l2_normalize_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        elements: u32, epsilon: f32,
    )
);

#[derive(Clone, Debug)]
pub struct L2NormalizeBf16 {
    kernel: TypedKernel<NormalizeKernel>,
    elements: usize,
}

impl L2NormalizeBf16 {
    pub fn compile(compiler: &Compiler, elements: usize) -> Result<Self> {
        if elements == 0 || elements > 65_536 {
            return Err(Error::InvalidDecoderKernel("invalid L2 normalization width"));
        }
        let source = cuda_kernel_file!("../../../kernels/text/normalize_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, elements })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if input.len() != self.elements || output.len() != self.elements {
            return Err(Error::InvalidDecoderKernel("L2 normalization buffer geometry differs"));
        }
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (1, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, output, u32::try_from(self.elements)?, 1.0e-12),
        )?)
    }
}
