use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{NvFp4Spec, geometry::narrow};
use crate::{Error, Result, kernels::geometry::require};

cuda_export!(
    NvFp4PrepareSelectedWeightKernel = "libmir_cuda_nvfp4_prepare_selected_weight"(
        source_weight: &DeviceBuffer<u8>,
        source_scales: &DeviceBuffer<u8>,
        source_input_scales: &DeviceBuffer<f32>,
        source_weight_scales: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<u32>,
        weight: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        input_scale: &mut DeviceBuffer<f32>,
        weight_scale: &mut DeviceBuffer<f32>,
        experts: u32,
        rank: u32,
        rows: u32,
        columns: u32,
    )
);

cuda_export!(
    NvFp4QuantizeSelectedKernel = "libmir_cuda_nvfp4_quantize_selected_bf16"(
        input: &DeviceBuffer<bf16>,
        input_offset: u32,
        global_scale: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        columns: u32,
    )
);

cuda_export!(
    NvFp4ScaleSelectedKernel = "libmir_cuda_nvfp4_scale_selected_bf16"(
        input: &DeviceBuffer<bf16>,
        input_scale: &DeviceBuffer<f32>,
        weight_scale: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
        output_offset: u32,
        elements: u32,
    )
);

#[derive(Clone, Debug)]
pub struct NvFp4SelectedWeightPreparation {
    kernel: TypedKernel<NvFp4PrepareSelectedWeightKernel>,
    scale: TypedKernel<NvFp4ScaleSelectedKernel>,
    quantize: TypedKernel<NvFp4QuantizeSelectedKernel>,
}

pub struct NvFp4SelectedWeightLaunch<'a> {
    pub source_weight: &'a DeviceBuffer<u8>,
    pub source_scales: &'a DeviceBuffer<u8>,
    pub source_input_scales: &'a DeviceBuffer<f32>,
    pub source_weight_scales: &'a DeviceBuffer<f32>,
    pub selected: &'a DeviceBuffer<u32>,
    pub rank: usize,
    pub weight: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub input_scale: &'a mut DeviceBuffer<f32>,
    pub weight_scale: &'a mut DeviceBuffer<f32>,
}

impl NvFp4SelectedWeightPreparation {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/nvfp4_prepare.cu");
        let options = CompileOptions { fast_math: false, ..Default::default() };
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            kernel: module.kernel()?,
            scale: module.kernel()?,
            quantize: module.kernel()?,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        spec: NvFp4Spec,
        experts: usize,
        launch: &mut NvFp4SelectedWeightLaunch<'_>,
    ) -> Result<()> {
        validate(spec, experts, launch)?;
        let packed = spec.elements()? / 2;
        let threads = 256_usize;
        let config = LaunchConfig {
            grid: (narrow(packed.div_ceil(threads))?, 1, 1),
            block: (narrow(threads)?, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.kernel.launch(
            stream,
            config,
            (
                launch.source_weight,
                launch.source_scales,
                launch.source_input_scales,
                launch.source_weight_scales,
                launch.selected,
                &mut *launch.weight,
                &mut *launch.scales,
                &mut *launch.input_scale,
                &mut *launch.weight_scale,
                narrow(experts)?,
                narrow(launch.rank)?,
                narrow(spec.output_features)?,
                narrow(spec.input_features)?,
            ),
        )?)
    }

    pub fn scale(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        input_scale: &DeviceBuffer<f32>,
        weight_scale: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
        output_offset: usize,
    ) -> Result<()> {
        require("NVFP4 selected input scale", 1, input_scale.len())?;
        require("NVFP4 selected weight scale", 1, weight_scale.len())?;
        let required = output_offset
            .checked_add(input.len())
            .ok_or(Error::InvalidNvFp4("selected output overflow"))?;
        require("NVFP4 selected output", required, output.len())?;
        let threads = 256_usize;
        let config = LaunchConfig {
            grid: (narrow(input.len().div_ceil(threads))?, 1, 1),
            block: (narrow(threads)?, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.scale.launch(
            stream,
            config,
            (
                input,
                input_scale,
                weight_scale,
                output,
                narrow(output_offset)?,
                narrow(input.len())?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn quantize(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        input_offset: usize,
        columns: usize,
        global_scale: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let required = input_offset
            .checked_add(columns)
            .ok_or(Error::InvalidNvFp4("selected input overflow"))?;
        require("NVFP4 selected input", required, input.len())?;
        require("NVFP4 selected global scale", 1, global_scale.len())?;
        require("NVFP4 selected packed input", columns / 2, packed.len())?;
        require("NVFP4 selected input scales", super::scale_elements(1, columns)?, scales.len())?;
        let config = LaunchConfig {
            grid: (narrow(columns / 16)?, 1, 1),
            block: (32, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.quantize.launch(
            stream,
            config,
            (input, narrow(input_offset)?, global_scale, packed, scales, narrow(columns)?),
        )?)
    }
}

fn validate(spec: NvFp4Spec, experts: usize, launch: &NvFp4SelectedWeightLaunch<'_>) -> Result<()> {
    if experts == 0 || launch.rank >= launch.selected.len() {
        return Err(Error::InvalidNvFp4("invalid selected expert geometry"));
    }
    let elements = spec.elements()?;
    let source_elements = elements
        .checked_mul(experts)
        .ok_or(Error::InvalidNvFp4("selected expert bank overflow"))?;
    require("NVFP4 expert weights", source_elements / 2, launch.source_weight.len())?;
    require("NVFP4 expert scales", source_elements / 16, launch.source_scales.len())?;
    require("NVFP4 expert input scales", experts, launch.source_input_scales.len())?;
    require("NVFP4 expert weight scales", experts, launch.source_weight_scales.len())?;
    require("NVFP4 selected weight", elements / 2, launch.weight.len())?;
    require("NVFP4 selected scales", spec.scale_elements()?, launch.scales.len())?;
    require("NVFP4 selected input scale", 1, launch.input_scale.len())?;
    require("NVFP4 selected weight scale", 1, launch.weight_scale.len())
}
