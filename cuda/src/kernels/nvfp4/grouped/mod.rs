use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::scale_elements;
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

mod gated;

cuda_export!(PrepareBankScalesKernel = "libmir_cuda_nvfp4_prepare_bank_scales"(
    source: &DeviceBuffer<u8>, output: &mut DeviceBuffer<u8>, experts: u32,
    rows: u32, columns: u32, output_stride: u32,
));
cuda_export!(QuantizeIndexedKernel = "libmir_cuda_nvfp4_quantize_indexed_bf16"(
    input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    global_scales: &DeviceBuffer<f32>, packed: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>, groups: u32, selected_count: u32,
    input_rows: u32, columns: u32, scale_stride: u32, ranked: u32,
));
cuda_export!(QuantizeIndexedPairKernel = "libmir_cuda_nvfp4_quantize_indexed_pair_bf16"(
    input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    left_globals: &DeviceBuffer<f32>, right_globals: &DeviceBuffer<f32>,
    left_packed: &mut DeviceBuffer<u8>, right_packed: &mut DeviceBuffer<u8>,
    left_scales: &mut DeviceBuffer<u8>, right_scales: &mut DeviceBuffer<u8>,
    groups: u32, selected_count: u32, input_rows: u32, columns: u32, scale_stride: u32,
));
cuda_export!(GatedQuantizeIndexedKernel = "libmir_cuda_nvfp4_gated_quantize_indexed_bf16"(
    gate: &DeviceBuffer<bf16>, up: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    global_scales: &DeviceBuffer<f32>, packed: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>, groups: u32, columns: u32,
    scale_stride: u32, activation: u32,
));

#[derive(Clone, Debug)]
pub struct NvFp4GroupedPreparation {
    prepare_scales: TypedKernel<PrepareBankScalesKernel>,
    quantize: TypedKernel<QuantizeIndexedKernel>,
    quantize_pair: TypedKernel<QuantizeIndexedPairKernel>,
    gated_quantize: TypedKernel<GatedQuantizeIndexedKernel>,
}

impl NvFp4GroupedPreparation {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../../kernels/nvfp4_grouped.cu");
        let options = CompileOptions { fast_math: false, ..Default::default() };
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            prepare_scales: module.kernel()?,
            quantize: module.kernel()?,
            quantize_pair: module.kernel()?,
            gated_quantize: module.kernel()?,
        })
    }

    pub fn prepare_bank_scales(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<u8>,
        geometry: BankScaleGeometry,
    ) -> Result<()> {
        let source_per_expert = product(geometry.rows, geometry.columns)? / 16;
        let output_stride = scale_elements(geometry.rows, geometry.columns)?;
        let source_elements = product(geometry.experts, source_per_expert)?;
        require("NVFP4 bank source scales", source_elements, source.len())?;
        require(
            "NVFP4 bank CUTLASS scales",
            product(geometry.experts, output_stride)?,
            output.len(),
        )?;
        let threads = 256_usize;
        Ok(self.prepare_scales.launch(
            stream,
            LaunchConfig {
                grid: (narrow(source_elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                source,
                output,
                narrow(geometry.experts)?,
                narrow(geometry.rows)?,
                narrow(geometry.columns)?,
                narrow(output_stride)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn quantize(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        global_scales: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        geometry: GroupedQuantize,
    ) -> Result<()> {
        geometry.validate(input, selected, global_scales, packed, scales)?;
        let blocks = product(geometry.groups, geometry.columns / 16)?;
        let scale_stride = scale_elements(1, geometry.columns)?;
        Ok(self.quantize.launch(
            stream,
            LaunchConfig {
                grid: (narrow(blocks)?, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                selected,
                global_scales,
                packed,
                scales,
                narrow(geometry.groups)?,
                narrow(geometry.selected)?,
                narrow(geometry.input_rows)?,
                narrow(geometry.columns)?,
                narrow(scale_stride)?,
                u32::from(geometry.ranked),
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn quantize_pair(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        left_globals: &DeviceBuffer<f32>,
        right_globals: &DeviceBuffer<f32>,
        left_packed: &mut DeviceBuffer<u8>,
        right_packed: &mut DeviceBuffer<u8>,
        left_scales: &mut DeviceBuffer<u8>,
        right_scales: &mut DeviceBuffer<u8>,
        geometry: GroupedQuantize,
    ) -> Result<()> {
        if geometry.ranked {
            return Err(Error::InvalidNvFp4("paired grouped quantization requires shared input"));
        }
        geometry.validate(input, selected, left_globals, left_packed, left_scales)?;
        geometry.validate(input, selected, right_globals, right_packed, right_scales)?;
        let blocks = product(geometry.groups, geometry.columns / 16)?;
        let scale_stride = scale_elements(1, geometry.columns)?;
        Ok(self.quantize_pair.launch(
            stream,
            LaunchConfig {
                grid: (narrow(blocks)?, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                selected,
                left_globals,
                right_globals,
                left_packed,
                right_packed,
                left_scales,
                right_scales,
                narrow(geometry.groups)?,
                narrow(geometry.selected)?,
                narrow(geometry.input_rows)?,
                narrow(geometry.columns)?,
                narrow(scale_stride)?,
            ),
        )?)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BankScaleGeometry {
    pub experts: usize,
    pub rows: usize,
    pub columns: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct GroupedQuantize {
    pub groups: usize,
    pub selected: usize,
    pub input_rows: usize,
    pub columns: usize,
    pub ranked: bool,
}

impl GroupedQuantize {
    fn validate(
        self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        globals: &DeviceBuffer<f32>,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
    ) -> Result<()> {
        if self.groups == 0 || self.selected == 0 || !self.columns.is_multiple_of(64) {
            return Err(Error::InvalidNvFp4("invalid grouped quantization geometry"));
        }
        require("grouped NVFP4 input", product(self.input_rows, self.columns)?, input.len())?;
        require("grouped NVFP4 indices", self.groups, selected.len())?;
        require("grouped NVFP4 globals", 1, globals.len())?;
        require("grouped NVFP4 packed", product(self.groups, self.columns / 2)?, packed.len())?;
        require(
            "grouped NVFP4 scales",
            product(self.groups, scale_elements(1, self.columns)?)?,
            scales.len(),
        )
    }
}
