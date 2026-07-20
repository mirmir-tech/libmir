use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    SigmoidMultiplyKernel = "libmir_cuda_sigmoid_multiply_bf16"(
        input: &DeviceBuffer<bf16>, gate: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
    )
);

cuda_export!(
    SigmoidElementwiseKernel = "libmir_cuda_sigmoid_multiply_elementwise_bf16"(
        input: &DeviceBuffer<bf16>, gate: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, elements: u32,
    )
);

#[derive(Clone, Debug)]
pub struct SigmoidMultiplyBf16 {
    kernel: TypedKernel<SigmoidMultiplyKernel>,
    rows: usize,
    columns: usize,
}

#[derive(Clone, Debug)]
pub struct SigmoidElementwiseBf16 {
    kernel: TypedKernel<SigmoidElementwiseKernel>,
    elements: usize,
}

impl SigmoidMultiplyBf16 {
    pub fn compile(compiler: &Compiler, rows: usize, columns: usize) -> Result<Self> {
        if rows == 0 || columns == 0 {
            return Err(Error::InvalidDecoderKernel("empty sigmoid multiply geometry"));
        }
        let source = cuda_kernel_file!("../../kernels/sigmoid_multiply_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, rows, columns })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(self.rows, self.columns)?;
        require("sigmoid multiply input", elements, input.len())?;
        require("sigmoid multiply gate", self.rows, gate.len())?;
        require("sigmoid multiply output", elements, output.len())?;
        let threads = 256_usize;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, gate, output, narrow(self.rows)?, narrow(self.columns)?),
        )?)
    }
}

impl SigmoidElementwiseBf16 {
    pub fn compile(compiler: &Compiler, elements: usize) -> Result<Self> {
        if elements == 0 {
            return Err(Error::InvalidDecoderKernel("empty elementwise sigmoid geometry"));
        }
        let source = cuda_kernel_file!("../../kernels/sigmoid_multiply_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, elements })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("elementwise sigmoid input", self.elements, input.len())?;
        require("elementwise sigmoid gate", self.elements, gate.len())?;
        require("elementwise sigmoid output", self.elements, output.len())?;
        let threads = 256_usize;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, gate, output, narrow(self.elements)?),
        )?)
    }
}
