use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{Error, Result};

cuda_export!(LogitSoftcapKernel = "libmir_cuda_logit_softcap_bf16"(
    values: &mut DeviceBuffer<bf16>, elements: u32, cap: f32,
));

#[derive(Clone, Debug)]
pub struct LogitSoftcap {
    kernel: TypedKernel<LogitSoftcapKernel>,
    elements: usize,
    cap: f32,
}

impl LogitSoftcap {
    pub(crate) fn compile(compiler: &Compiler, elements: usize, cap: f32) -> Result<Self> {
        if elements == 0 || !cap.is_finite() || cap <= 0.0 {
            return Err(Error::InvalidDecoderKernel("invalid logit softcap configuration"));
        }
        let source = cuda_kernel_file!("../../kernels/logit_softcap_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, elements, cap })
    }

    pub(crate) fn execute(&self, stream: &Stream, values: &mut DeviceBuffer<bf16>) -> Result<()> {
        if values.len() != self.elements {
            return Err(Error::InvalidDecoderKernel("logit softcap buffer width differs"));
        }
        let threads = 256_usize;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(self.elements.div_ceil(threads))?, 1, 1),
                block: (u32::try_from(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (values, u32::try_from(self.elements)?, self.cap),
        )?)
    }
}
