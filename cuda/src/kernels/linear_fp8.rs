use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(QuantizeBlockFp8Weight = "libmir_cuda_quantize_block_fp8_weight"(
    source: &DeviceBuffer<bf16>, weight: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<f32>, rows: u32, columns: u32,
));
cuda_export!(BlockFp8LinearKernel = "libmir_cuda_block_fp8x4_bf16_linear"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>, scales: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
));

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlockFp8LinearSpec {
    pub input_features: usize,
    pub output_features: usize,
}

impl BlockFp8LinearSpec {
    pub fn new(input_features: usize, output_features: usize) -> Result<Self> {
        if input_features == 0
            || output_features == 0
            || !input_features.is_multiple_of(128)
            || !output_features.is_multiple_of(128)
        {
            return Err(Error::InvalidDecoderKernel("invalid block FP8 linear geometry"));
        }
        let _ = product(input_features, output_features)?;
        Ok(Self { input_features, output_features })
    }

    pub fn weight_elements(self) -> Result<usize> {
        product(self.input_features, self.output_features)
    }

    pub fn scale_elements(self) -> Result<usize> {
        product(self.output_features, self.input_features / 128)
    }
}

#[derive(Clone, Debug)]
pub struct BlockFp8LinearKernels {
    quantize: TypedKernel<QuantizeBlockFp8Weight>,
    project: TypedKernel<BlockFp8LinearKernel>,
    spec: BlockFp8LinearSpec,
}

impl BlockFp8LinearKernels {
    pub fn compile(compiler: &Compiler, spec: BlockFp8LinearSpec) -> Result<Self> {
        let source = cuda_kernel_file!("../../kernels/linear_fp8_block.cu");
        let module = compiler.compile(
            source,
            &CompileOptions {
                fast_math: false,
                ..CompileOptions::default()
            },
        )?;
        Ok(Self {
            quantize: module.kernel()?,
            project: module.kernel()?,
            spec,
        })
    }

    pub fn quantize(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<bf16>,
        weight: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.validate_weight(weight, scales)?;
        require("block FP8 source", self.spec.weight_elements()?, source.len())?;
        Ok(self.quantize.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(self.spec.output_features)?,
                    narrow(self.spec.input_features / 128)?,
                    1,
                ),
                block: (128, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                source,
                weight,
                scales,
                narrow(self.spec.output_features)?,
                narrow(self.spec.input_features)?,
            ),
        )?)
    }

    pub fn project(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("block FP8 input", self.spec.input_features, input.len())?;
        self.validate_weight(weight, scales)?;
        require("block FP8 output", self.spec.output_features, output.len())?;
        Ok(self.project.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.output_features.div_ceil(64))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weight,
                scales,
                output,
                narrow(self.spec.output_features)?,
                narrow(self.spec.input_features)?,
            ),
        )?)
    }

    fn validate_weight(&self, weight: &DeviceBuffer<u8>, scales: &DeviceBuffer<f32>) -> Result<()> {
        require("block FP8 weight", self.spec.weight_elements()?, weight.len())?;
        require("block FP8 scales", self.spec.scale_elements()?, scales.len())
    }
}
