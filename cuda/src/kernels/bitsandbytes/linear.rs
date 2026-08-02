use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(Bnb4GemvKernel = "libmir_cuda_bnb4_gemv_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>, absmax: &DeviceBuffer<u8>,
    quant_map: &DeviceBuffer<u8>, nested_absmax: &DeviceBuffer<u8>,
    nested_quant_map: &DeviceBuffer<u8>, output: &mut DeviceBuffer<bf16>,
    input_features: u32, output_features: u32, block_size: u32, nested_block_size: u32,
    nested_offset: f32,
));

cuda_export!(Bnb4QmmKernel = "libmir_cuda_bnb4_qmm_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>, absmax: &DeviceBuffer<u8>,
    quant_map: &DeviceBuffer<u8>, nested_absmax: &DeviceBuffer<u8>,
    nested_quant_map: &DeviceBuffer<u8>, output: &mut DeviceBuffer<bf16>, tokens: u32,
    input_features: u32, output_features: u32, block_size: u32, nested_block_size: u32,
    nested_offset: f32,
));

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BitsAndBytes4BitSpec {
    tokens: usize,
    input: usize,
    output: usize,
    block_size: usize,
    nested_block_size: Option<usize>,
}

impl BitsAndBytes4BitSpec {
    pub const fn new(
        tokens: usize,
        input: usize,
        output: usize,
        block_size: usize,
        nested_block_size: Option<usize>,
    ) -> Result<Self> {
        if tokens == 0
            || input == 0
            || output == 0
            || block_size != 64
            || !input.is_multiple_of(16)
            || !matches!(nested_block_size, None | Some(256))
        {
            return Err(Error::InvalidQuantizedGemv("unsupported bitsandbytes geometry"));
        }
        Ok(Self {
            tokens,
            input,
            output,
            block_size,
            nested_block_size,
        })
    }
}

#[derive(Clone, Debug)]
pub struct BitsAndBytes4BitLinear {
    kernel: Kernel,
    spec: BitsAndBytes4BitSpec,
}

#[derive(Clone, Debug)]
enum Kernel {
    Gemv(TypedKernel<Bnb4GemvKernel>),
    Qmm(TypedKernel<Bnb4QmmKernel>),
}

pub struct BitsAndBytes4BitLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub weight: &'a DeviceBuffer<u8>,
    pub absmax: &'a DeviceBuffer<u8>,
    pub quant_map: &'a DeviceBuffer<u8>,
    pub nested_absmax: &'a DeviceBuffer<u8>,
    pub nested_quant_map: &'a DeviceBuffer<u8>,
    pub nested_offset: f32,
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl BitsAndBytes4BitLinear {
    pub fn compile(compiler: &Compiler, spec: BitsAndBytes4BitSpec) -> Result<Self> {
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/bnb4_bf16.cu"),
            &CompileOptions { fast_math: true, ..Default::default() },
        )?;
        let kernel = if spec.tokens == 1 {
            Kernel::Gemv(module.kernel()?)
        } else {
            Kernel::Qmm(module.kernel()?)
        };
        Ok(Self { kernel, spec })
    }

    pub fn execute(&self, stream: &Stream, launch: &mut BitsAndBytes4BitLaunch<'_>) -> Result<()> {
        require("bnb4 input", product(self.spec.tokens, self.spec.input)?, launch.input.len())?;
        require(
            "bnb4 weight",
            product(self.spec.output, self.spec.input.div_ceil(2))?,
            launch.weight.len(),
        )?;
        require("bnb4 output", product(self.spec.tokens, self.spec.output)?, launch.output.len())?;
        match &self.kernel {
            Kernel::Gemv(kernel) => self.launch_gemv(kernel, stream, launch),
            Kernel::Qmm(kernel) => self.launch_qmm(kernel, stream, launch),
        }
    }

    fn launch_gemv(
        &self,
        kernel: &TypedKernel<Bnb4GemvKernel>,
        stream: &Stream,
        launch: &mut BitsAndBytes4BitLaunch<'_>,
    ) -> Result<()> {
        Ok(kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.output.div_ceil(8))?, 1, 1),
                block: (32, 8, 1),
                shared_memory_bytes: 0,
            },
            (
                launch.input,
                launch.weight,
                launch.absmax,
                launch.quant_map,
                launch.nested_absmax,
                launch.nested_quant_map,
                &mut *launch.output,
                narrow(self.spec.input)?,
                narrow(self.spec.output)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.nested_block_size.unwrap_or(0))?,
                launch.nested_offset,
            ),
        )?)
    }

    fn launch_qmm(
        &self,
        kernel: &TypedKernel<Bnb4QmmKernel>,
        stream: &Stream,
        launch: &mut BitsAndBytes4BitLaunch<'_>,
    ) -> Result<()> {
        Ok(kernel.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(self.spec.output.div_ceil(16))?,
                    narrow(self.spec.tokens.div_ceil(64))?,
                    1,
                ),
                block: (32, 4, 1),
                shared_memory_bytes: 0,
            },
            (
                launch.input,
                launch.weight,
                launch.absmax,
                launch.quant_map,
                launch.nested_absmax,
                launch.nested_quant_map,
                &mut *launch.output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.input)?,
                narrow(self.spec.output)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.nested_block_size.unwrap_or(0))?,
                launch.nested_offset,
            ),
        )?)
    }
}
