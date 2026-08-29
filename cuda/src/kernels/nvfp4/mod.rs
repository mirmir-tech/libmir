use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

mod buckets;
mod gated;
mod grouped;
mod micro;
mod norm;
mod selected;
mod weight_only;
mod weight_only_tensor_core;
pub use buckets::{BucketGeometry, BucketQuantize, NvFp4BucketPreparation};
pub use gated::NvFp4Gated;
pub use grouped::{BankScaleGeometry, GroupedQuantize, NvFp4GroupedPreparation};
pub use micro::{
    NvFp4MicroBanks, NvFp4MicroDownKernels, NvFp4MicroDownLaunch, NvFp4MicroDownWorkspace,
    NvFp4MicroGateLaunch, NvFp4MicroGateWorkspace, NvFp4MicroKernels, NvFp4MicroLaunch,
    NvFp4MicroSpec, NvFp4MicroWorkspace,
};
pub use norm::NvFp4RmsNorm;
pub use selected::{NvFp4SelectedWeightLaunch, NvFp4SelectedWeightPreparation};
pub use weight_only::{NvFp4WeightOnly, NvFp4WeightOnlyLaunch};
pub use weight_only_tensor_core::NvFp4WeightOnlyTensorCore;

cuda_export!(
    NvFp4DequantKernel = "libmir_cuda_nvfp4_dequant_bf16"(
        packed: &DeviceBuffer<u8>,
        block_scales: &DeviceBuffer<u8>,
        global_scale: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
        elements: u32,
    )
);

cuda_export!(
    NvFp4PrepareWeightKernel = "libmir_cuda_nvfp4_prepare_weight"(
        source_weight: &DeviceBuffer<u8>,
        source_scales: &DeviceBuffer<u8>,
        weight: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        rows: u32,
        columns: u32,
    )
);

cuda_export!(
    NvFp4QuantizeKernel = "libmir_cuda_nvfp4_quantize_bf16"(
        input: &DeviceBuffer<bf16>,
        global_scale: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        rows: u32,
        columns: u32,
    )
);

/// Matrix geometry for NVIDIA E2M1 weights with E4M3 block scaling.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NvFp4Spec {
    pub input_features: usize,
    pub output_features: usize,
}

impl NvFp4Spec {
    /// Creates one NVFP4 matrix contract with fixed 16-value blocks.
    pub fn new(input_features: usize, output_features: usize) -> Result<Self> {
        if input_features == 0 || output_features == 0 {
            return Err(Error::InvalidNvFp4("dimensions must be non-zero"));
        }
        if !input_features.is_multiple_of(16) {
            return Err(Error::InvalidNvFp4("input features must align to 16-value blocks"));
        }
        Ok(Self { input_features, output_features })
    }

    pub fn elements(self) -> Result<usize> {
        product(self.input_features, self.output_features)
    }

    pub fn scale_elements(self) -> Result<usize> {
        scale_elements(self.output_features, self.input_features)
    }
}

pub fn scale_elements(rows: usize, columns: usize) -> Result<usize> {
    rows.div_ceil(128)
        .checked_mul(columns / 64)
        .and_then(|tiles| tiles.checked_mul(512))
        .ok_or(Error::InvalidNvFp4("scale layout overflow"))
}

#[derive(Clone, Debug)]
pub struct NvFp4Preparation {
    prepare_weight: TypedKernel<NvFp4PrepareWeightKernel>,
    quantize: TypedKernel<NvFp4QuantizeKernel>,
}

impl NvFp4Preparation {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/nvfp4_prepare.cu");
        let options = CompileOptions { fast_math: false, ..Default::default() };
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            prepare_weight: module.kernel()?,
            quantize: module.kernel()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_weight(
        &self,
        stream: &Stream,
        spec: NvFp4Spec,
        source_weight: &DeviceBuffer<u8>,
        source_scales: &DeviceBuffer<u8>,
        weight: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let elements = spec.elements()?;
        require("NVFP4 source weight", elements / 2, source_weight.len())?;
        require("NVFP4 source scales", elements / 16, source_scales.len())?;
        require("NVFP4 prepared weight", elements / 2, weight.len())?;
        require("NVFP4 prepared scales", spec.scale_elements()?, scales.len())?;
        let threads = 256_u32;
        let config = linear_config(elements / 2, threads)?;
        Ok(self.prepare_weight.launch(
            stream,
            config,
            (
                source_weight,
                source_scales,
                weight,
                scales,
                narrow(spec.output_features)?,
                narrow(spec.input_features)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn quantize(
        &self,
        stream: &Stream,
        rows: usize,
        columns: usize,
        input: &DeviceBuffer<bf16>,
        global_scale: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        const WARPS_PER_BLOCK: usize = 8;
        let elements = product(rows, columns)?;
        require("NVFP4 input", elements, input.len())?;
        require("NVFP4 input global scale", 1, global_scale.len())?;
        require("NVFP4 packed input", elements / 2, packed.len())?;
        require("NVFP4 input scales", scale_elements(rows, columns)?, scales.len())?;
        let warps = (elements / 16).div_ceil(2);
        let config = LaunchConfig {
            grid: (narrow(warps.div_ceil(WARPS_PER_BLOCK))?, 1, 1),
            block: (narrow(WARPS_PER_BLOCK * 32)?, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.quantize.launch(
            stream,
            config,
            (input, global_scale, packed, scales, narrow(rows)?, narrow(columns)?),
        )?)
    }
}

fn linear_config(elements: usize, threads: u32) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(usize::try_from(threads)?))?, 1, 1),
        block: (threads, 1, 1),
        shared_memory_bytes: 0,
    })
}

/// Compiled checkpoint dequantization used as the NVFP4 quality reference.
#[derive(Clone, Debug)]
pub struct NvFp4Dequant {
    kernel: TypedKernel<NvFp4DequantKernel>,
    spec: NvFp4Spec,
}

pub struct NvFp4DequantLaunch<'a> {
    pub packed: &'a DeviceBuffer<u8>,
    pub block_scales: &'a DeviceBuffer<u8>,
    pub global_scale: &'a DeviceBuffer<f32>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl NvFp4Dequant {
    pub fn compile(compiler: &Compiler, spec: NvFp4Spec) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/nvfp4_dequant_bf16.cu");
        let options = CompileOptions { fast_math: false, ..Default::default() };
        let module = compiler.compile(source, &options)?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(&self, stream: &Stream, launch: &mut NvFp4DequantLaunch<'_>) -> Result<()> {
        let elements = self.spec.elements()?;
        require("NVFP4 packed weight", elements / 2, launch.packed.len())?;
        require("NVFP4 block scales", elements / 16, launch.block_scales.len())?;
        require("NVFP4 global scale", 1, launch.global_scale.len())?;
        require("NVFP4 BF16 output", elements, launch.output.len())?;
        let threads = 256_u32;
        let config = LaunchConfig {
            grid: (narrow(elements.div_ceil(usize::try_from(threads)?))?, 1, 1),
            block: (threads, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.kernel.launch(
            stream,
            config,
            (
                launch.packed,
                launch.block_scales,
                launch.global_scale,
                &mut *launch.output,
                narrow(elements)?,
            ),
        )?)
    }
}
