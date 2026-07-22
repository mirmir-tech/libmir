use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(PackedInt8GemvKernel = "libmir_cuda_packed_int8_gemv_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<i32>, scales: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>, input_features: u32, output_features: u32,
));

cuda_export!(PackedInt8QmmKernel = "libmir_cuda_packed_int8_qmm_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<i32>, scales: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>, tokens: u32, input_features: u32, output_features: u32,
));

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PackedInt8Spec {
    pub tokens: usize,
    pub input_features: usize,
    pub output_features: usize,
}

impl PackedInt8Spec {
    pub const fn new(tokens: usize, input_features: usize, output_features: usize) -> Result<Self> {
        if tokens == 0 || input_features == 0 || output_features == 0 {
            return Err(Error::InvalidQuantizedGemv("packed INT8 dimensions must be non-zero"));
        }
        if !input_features.is_multiple_of(16) {
            return Err(Error::InvalidQuantizedGemv(
                "packed INT8 input must align to sixteen features",
            ));
        }
        Ok(Self { tokens, input_features, output_features })
    }

    fn packed_elements(self) -> Result<usize> {
        product(self.output_features, self.input_features / 4)
    }
}

#[derive(Clone, Debug)]
pub struct PackedInt8Linear {
    kernel: PackedInt8Kernel,
    spec: PackedInt8Spec,
}

#[derive(Clone, Debug)]
enum PackedInt8Kernel {
    Gemv(TypedKernel<PackedInt8GemvKernel>),
    Qmm(TypedKernel<PackedInt8QmmKernel>),
}

pub struct PackedInt8Launch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub weight: &'a DeviceBuffer<i32>,
    pub scales: &'a DeviceBuffer<bf16>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl PackedInt8Linear {
    pub fn compile(compiler: &Compiler, spec: PackedInt8Spec) -> Result<Self> {
        let source = cuda_kernel_file!("../../kernels/packed_int8_bf16.cu");
        let module =
            compiler.compile(source, &CompileOptions { fast_math: true, ..Default::default() })?;
        let kernel = if spec.tokens == 1 {
            PackedInt8Kernel::Gemv(module.kernel()?)
        } else {
            PackedInt8Kernel::Qmm(module.kernel()?)
        };
        Ok(Self { kernel, spec })
    }

    pub fn execute(&self, stream: &Stream, launch: &mut PackedInt8Launch<'_>) -> Result<()> {
        require(
            "packed INT8 input",
            product(self.spec.tokens, self.spec.input_features)?,
            launch.input.len(),
        )?;
        require("packed INT8 weight", self.spec.packed_elements()?, launch.weight.len())?;
        require("packed INT8 scales", self.spec.output_features, launch.scales.len())?;
        require(
            "packed INT8 output",
            product(self.spec.tokens, self.spec.output_features)?,
            launch.output.len(),
        )?;
        match &self.kernel {
            PackedInt8Kernel::Gemv(kernel) => Ok(kernel.launch(
                stream,
                LaunchConfig {
                    grid: (narrow(self.spec.output_features.div_ceil(8))?, 1, 1),
                    block: (32, 8, 1),
                    shared_memory_bytes: 0,
                },
                (
                    launch.input,
                    launch.weight,
                    launch.scales,
                    &mut *launch.output,
                    narrow(self.spec.input_features)?,
                    narrow(self.spec.output_features)?,
                ),
            )?),
            PackedInt8Kernel::Qmm(kernel) => Ok(kernel.launch(
                stream,
                LaunchConfig {
                    grid: (
                        narrow(self.spec.output_features.div_ceil(16))?,
                        narrow(self.spec.tokens.div_ceil(64))?,
                        1,
                    ),
                    block: (32, 4, 1),
                    shared_memory_bytes: 0,
                },
                (
                    launch.input,
                    launch.weight,
                    launch.scales,
                    &mut *launch.output,
                    narrow(self.spec.tokens)?,
                    narrow(self.spec.input_features)?,
                    narrow(self.spec.output_features)?,
                ),
            )?),
        }
    }
}
