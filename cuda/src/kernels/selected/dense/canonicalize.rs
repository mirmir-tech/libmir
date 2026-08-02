use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{Error, Result, kernels::geometry::narrow};

cuda_export!(CanonicalizeKernel = "libmir_cuda_dense_expert_canonicalize_bf16"(
    input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    matrices: u32, input_features: u32, output_features: u32,
));

#[derive(Clone, Debug)]
pub struct DenseExpertCanonicalizer {
    kernel: TypedKernel<CanonicalizeKernel>,
}

impl DenseExpertCanonicalizer {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../../kernels/selected_dense_moe_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()? })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        matrices: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<()> {
        let elements = matrices
            .checked_mul(input_features)
            .and_then(|value| value.checked_mul(output_features))
            .ok_or(Error::InvalidDecoderKernel("dense expert transpose size overflow"))?;
        if input.len() != elements || output.len() != elements {
            return Err(Error::InvalidDecoderKernel(
                "dense expert transpose buffers differ from geometry",
            ));
        }
        Ok(self.kernel.launch(
            stream,
            mircuda::LaunchConfig {
                grid: (narrow(elements.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                output,
                narrow(matrices)?,
                narrow(input_features)?,
                narrow(output_features)?,
            ),
        )?)
    }
}
