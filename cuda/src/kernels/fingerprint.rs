use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{Error, Result};

cuda_export!(
    ActivationFingerprintKernel = "libmir_cuda_activation_fingerprint_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<u64>,
        elements: u32, layer: u32,
    )
);

pub struct ActivationFingerprint {
    kernel: TypedKernel<ActivationFingerprintKernel>,
    elements: usize,
}

impl ActivationFingerprint {
    pub(crate) fn compile(compiler: &Compiler, elements: usize) -> Result<Self> {
        if elements == 0 {
            return Err(Error::InvalidDecoderKernel("activation fingerprint has no elements"));
        }
        let source = cuda_kernel_file!("../../kernels/activation_fingerprint_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, elements })
    }

    pub(crate) fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<u64>,
        layer: usize,
    ) -> Result<()> {
        if input.len() != self.elements || output.len() < (layer + 1) * 2 {
            return Err(Error::InvalidDecoderKernel("activation fingerprint geometry differs"));
        }
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (1, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, output, u32::try_from(self.elements)?, u32::try_from(layer)?),
        )?)
    }
}
