use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file, f16,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(AwqGemvKernel = "libmir_cuda_awq_gemv_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<i32>,
    zero_points: &DeviceBuffer<i32>, scales: &DeviceBuffer<f16>,
    output: &mut DeviceBuffer<bf16>, input_features: u32, output_features: u32, group_size: u32,
));

cuda_export!(AwqQmmKernel = "libmir_cuda_awq_qmm_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<i32>,
    zero_points: &DeviceBuffer<i32>, scales: &DeviceBuffer<f16>,
    output: &mut DeviceBuffer<bf16>, tokens: u32, input_features: u32, output_features: u32,
    group_size: u32,
));

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AwqSpec {
    pub tokens: usize,
    pub input_features: usize,
    pub output_features: usize,
    pub group_size: usize,
}

impl AwqSpec {
    pub const fn new(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        group_size: usize,
    ) -> Result<Self> {
        if tokens == 0
            || input_features == 0
            || output_features == 0
            || group_size == 0
            || !input_features.is_multiple_of(16)
            || !input_features.is_multiple_of(group_size)
            || !output_features.is_multiple_of(8)
        {
            return Err(Error::InvalidQuantizedGemv("AWQ dimensions are unsupported"));
        }
        Ok(Self {
            tokens,
            input_features,
            output_features,
            group_size,
        })
    }

    fn packed_output(self) -> usize {
        self.output_features / 8
    }
}

pub struct AwqLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub weight: &'a DeviceBuffer<i32>,
    pub zero_points: &'a DeviceBuffer<i32>,
    pub scales: &'a DeviceBuffer<f16>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

#[derive(Clone, Debug)]
pub struct AwqLinear {
    kernel: AwqKernel,
    spec: AwqSpec,
}

#[derive(Clone, Debug)]
enum AwqKernel {
    Gemv(TypedKernel<AwqGemvKernel>),
    Qmm(TypedKernel<AwqQmmKernel>),
}

impl AwqLinear {
    pub fn compile(compiler: &Compiler, spec: AwqSpec) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/awq_bf16.cu");
        let module =
            compiler.compile(source, &CompileOptions { fast_math: true, ..Default::default() })?;
        let kernel = if spec.tokens == 1 {
            AwqKernel::Gemv(module.kernel()?)
        } else {
            AwqKernel::Qmm(module.kernel()?)
        };
        Ok(Self { kernel, spec })
    }

    pub fn execute(&self, stream: &Stream, launch: &mut AwqLaunch<'_>) -> Result<()> {
        let groups = self.spec.input_features / self.spec.group_size;
        require(
            "AWQ input",
            product(self.spec.tokens, self.spec.input_features)?,
            launch.input.len(),
        )?;
        require(
            "AWQ weight",
            product(self.spec.input_features, self.spec.packed_output())?,
            launch.weight.len(),
        )?;
        require(
            "AWQ zero points",
            product(groups, self.spec.packed_output())?,
            launch.zero_points.len(),
        )?;
        require("AWQ scales", product(groups, self.spec.output_features)?, launch.scales.len())?;
        require(
            "AWQ output",
            product(self.spec.tokens, self.spec.output_features)?,
            launch.output.len(),
        )?;
        match &self.kernel {
            AwqKernel::Gemv(kernel) => self.launch_gemv(kernel, stream, launch),
            AwqKernel::Qmm(kernel) => self.launch_qmm(kernel, stream, launch),
        }
    }

    fn launch_gemv(
        &self,
        kernel: &TypedKernel<AwqGemvKernel>,
        stream: &Stream,
        launch: &mut AwqLaunch<'_>,
    ) -> Result<()> {
        Ok(kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.output_features.div_ceil(8))?, 1, 1),
                block: (32, 8, 1),
                shared_memory_bytes: 0,
            },
            (
                launch.input,
                launch.weight,
                launch.zero_points,
                launch.scales,
                &mut *launch.output,
                narrow(self.spec.input_features)?,
                narrow(self.spec.output_features)?,
                narrow(self.spec.group_size)?,
            ),
        )?)
    }

    fn launch_qmm(
        &self,
        kernel: &TypedKernel<AwqQmmKernel>,
        stream: &Stream,
        launch: &mut AwqLaunch<'_>,
    ) -> Result<()> {
        Ok(kernel.launch(
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
                launch.zero_points,
                launch.scales,
                &mut *launch.output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.input_features)?,
                narrow(self.spec.output_features)?,
                narrow(self.spec.group_size)?,
            ),
        )?)
    }
}
