use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

mod refine;
mod residual;

pub use refine::Fp8RefinementKernels;

cuda_export!(QuantizeWeightKernel = "libmir_cuda_output_quantize_fp8_weight"(
    source: &DeviceBuffer<bf16>, weight: &mut DeviceBuffer<u8>, scales: &mut DeviceBuffer<f32>,
    row_scales: &mut DeviceBuffer<f32>, rows: u32, columns: u32,
));
cuda_export!(QuantizeInputKernel = "libmir_cuda_output_quantize_fp8_input"(
    source: &DeviceBuffer<bf16>, input: &mut DeviceBuffer<u8>, scales: &mut DeviceBuffer<f32>,
    columns: u32,
));
cuda_export!(RescaleKernel = "libmir_cuda_output_rescale_fp8_bf16"(
    output: &mut DeviceBuffer<bf16>, row_scales: &DeviceBuffer<f32>, rows: u32,
));
cuda_export!(VectorKernel = "libmir_cuda_output_fp8x4_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>, row_scales: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
));
cuda_export!(QuantizeResidualKernel = "libmir_cuda_output_quantize_fp8_int4_weight"(
    source: &DeviceBuffer<bf16>, weight: &mut DeviceBuffer<u8>,
    row_scales: &mut DeviceBuffer<f32>, residual: &mut DeviceBuffer<u8>,
    residual_scales: &mut DeviceBuffer<f32>, rows: u32, columns: u32,
));
cuda_export!(ResidualKernel = "libmir_cuda_output_fp8_int4_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>, row_scales: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<u8>, residual_scales: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
));

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Fp8OutputSpec {
    pub input_features: usize,
    pub output_features: usize,
}

/// Mutable storage populated while preparing an FP8 plus INT4 residual weight.
#[derive(Debug)]
pub struct Fp8ResidualWeightBuffers<'a> {
    weight: &'a mut DeviceBuffer<u8>,
    block_scales: &'a mut DeviceBuffer<f32>,
    row_scales: &'a mut DeviceBuffer<f32>,
    residual: &'a mut DeviceBuffer<u8>,
    residual_scales: &'a mut DeviceBuffer<f32>,
}

impl<'a> Fp8ResidualWeightBuffers<'a> {
    pub const fn new(
        weight: &'a mut DeviceBuffer<u8>,
        block_scales: &'a mut DeviceBuffer<f32>,
        row_scales: &'a mut DeviceBuffer<f32>,
        residual: &'a mut DeviceBuffer<u8>,
        residual_scales: &'a mut DeviceBuffer<f32>,
    ) -> Self {
        Self {
            weight,
            block_scales,
            row_scales,
            residual,
            residual_scales,
        }
    }
}

impl Fp8OutputSpec {
    pub fn new(input_features: usize, output_features: usize) -> Result<Self> {
        if input_features == 0
            || output_features == 0
            || !input_features.is_multiple_of(128)
            || !output_features.is_multiple_of(128)
        {
            return Err(Error::InvalidDecoderKernel("invalid blockwise FP8 output geometry"));
        }
        let _ = product(input_features, output_features)?;
        Ok(Self { input_features, output_features })
    }

    pub fn weight_elements(self) -> Result<usize> {
        product(self.input_features, self.output_features)
    }

    #[must_use]
    pub const fn input_scale_elements(self) -> usize {
        self.input_features / 128
    }

    pub fn weight_scale_elements(self) -> Result<usize> {
        product(self.input_features / 128, self.output_features / 128)
    }

    pub fn residual_elements(self) -> Result<usize> {
        self.weight_elements().map(|elements| elements / 2)
    }

    pub fn residual_scale_elements(self) -> Result<usize> {
        product(self.output_features, self.input_features / 128)
    }
}

#[derive(Clone, Debug)]
pub struct Fp8OutputKernels {
    weight: TypedKernel<QuantizeWeightKernel>,
    input: TypedKernel<QuantizeInputKernel>,
    rescale: TypedKernel<RescaleKernel>,
    vector: TypedKernel<VectorKernel>,
    quantize_residual: TypedKernel<QuantizeResidualKernel>,
    residual: TypedKernel<ResidualKernel>,
    spec: Fp8OutputSpec,
}

impl Fp8OutputKernels {
    pub fn compile(compiler: &Compiler, spec: Fp8OutputSpec) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/output_fp8.cu");
        let options = CompileOptions {
            fast_math: false,
            ..CompileOptions::default()
        };
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            weight: module.kernel()?,
            input: module.kernel()?,
            rescale: module.kernel()?,
            vector: module.kernel()?,
            quantize_residual: module.kernel()?,
            residual: module.kernel()?,
            spec,
        })
    }

    pub fn quantize_weight(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<bf16>,
        weight: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<f32>,
        row_scales: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        require("FP8 output source", self.spec.weight_elements()?, source.len())?;
        require("FP8 output weight", self.spec.weight_elements()?, weight.len())?;
        require("FP8 output scales", self.spec.weight_scale_elements()?, scales.len())?;
        require("FP8 output row scales", self.spec.output_features, row_scales.len())?;
        Ok(self.weight.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.output_features)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                source,
                weight,
                scales,
                row_scales,
                narrow(self.spec.output_features)?,
                narrow(self.spec.input_features)?,
            ),
        )?)
    }

    pub fn quantize_input(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<bf16>,
        input: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        require("FP8 output input source", self.spec.input_features, source.len())?;
        require("FP8 output input", self.spec.input_features, input.len())?;
        require("FP8 output input scales", self.spec.input_scale_elements(), scales.len())?;
        Ok(self.input.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.input_scale_elements())?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (source, input, scales, narrow(self.spec.input_features)?),
        )?)
    }

    pub fn rescale_output(
        &self,
        stream: &Stream,
        output: &mut DeviceBuffer<bf16>,
        row_scales: &DeviceBuffer<f32>,
    ) -> Result<()> {
        require("FP8 output logits", self.spec.output_features, output.len())?;
        require("FP8 output row scales", self.spec.output_features, row_scales.len())?;
        Ok(self.rescale.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.output_features.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (output, row_scales, narrow(self.spec.output_features)?),
        )?)
    }

    pub fn project_vectorized(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        row_scales: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("FP8 output input", self.spec.input_features, input.len())?;
        require("FP8 output weight", self.spec.weight_elements()?, weight.len())?;
        require("FP8 output row scales", self.spec.output_features, row_scales.len())?;
        require("FP8 output logits", self.spec.output_features, output.len())?;
        Ok(self.vector.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.output_features.div_ceil(64))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weight,
                row_scales,
                output,
                narrow(self.spec.output_features)?,
                narrow(self.spec.input_features)?,
            ),
        )?)
    }
}
