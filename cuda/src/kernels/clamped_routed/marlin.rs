use mircuda::{
    DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::math::narrow;
use crate::{Error, Result};

cuda_export!(GateUpKernel = "libmir_cuda_clamped_routed_marlin_gate_up_bf16"(
    input: &DeviceBuffer<bf16>, bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, assignments: u32, intermediate: u32,
    padded_intermediate: u32, limit: f32,
));
cuda_export!(DownReduceKernel = "libmir_cuda_clamped_routed_marlin_down_reduce_bf16"(
    input: &DeviceBuffer<bf16>, bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, tokens: u32, top_k: u32,
    hidden: u32, padded_hidden: u32,
));
cuda_export!(PadRowsKernel = "libmir_cuda_clamped_routed_marlin_pad_rows_bf16"(
    input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
    padded_columns: u32,
));

#[derive(Clone, Debug)]
pub struct ClampedRoutedMarlinEpilogue {
    gate_up: TypedKernel<GateUpKernel>,
    down_reduce: TypedKernel<DownReduceKernel>,
    pad_rows: TypedKernel<PadRowsKernel>,
    assignments: usize,
    tokens: usize,
    top_k: usize,
    intermediate: usize,
    hidden: usize,
    padded_hidden: usize,
    padded_intermediate: usize,
    limit: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ClampedRoutedMarlinGeometry {
    pub tokens: usize,
    pub top_k: usize,
    pub intermediate: usize,
    pub hidden: usize,
    pub padded_hidden: usize,
    pub padded_intermediate: usize,
    pub limit: f32,
}

impl ClampedRoutedMarlinEpilogue {
    pub fn compile(
        compiler: &mircuda::Compiler,
        geometry: ClampedRoutedMarlinGeometry,
    ) -> Result<Self> {
        let assignments = geometry
            .tokens
            .checked_mul(geometry.top_k)
            .ok_or(Error::InvalidDecoderKernel("clamped Marlin assignment overflow"))?;
        if geometry.tokens == 0
            || geometry.top_k == 0
            || geometry.intermediate == 0
            || geometry.hidden == 0
            || geometry.padded_hidden < geometry.hidden
            || geometry.padded_intermediate < geometry.intermediate
            || !geometry.limit.is_finite()
        {
            return Err(Error::InvalidDecoderKernel("invalid clamped Marlin epilogue geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/clamped_routed_marlin_bf16.cu"),
            &mircuda::CompileOptions::default(),
        )?;
        Ok(Self {
            gate_up: module.kernel()?,
            down_reduce: module.kernel()?,
            pad_rows: module.kernel()?,
            assignments,
            tokens: geometry.tokens,
            top_k: geometry.top_k,
            intermediate: geometry.intermediate,
            hidden: geometry.hidden,
            padded_hidden: geometry.padded_hidden,
            padded_intermediate: geometry.padded_intermediate,
            limit: geometry.limit,
        })
    }

    pub fn pad_input(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
    ) -> Result<()> {
        self.pad(stream, input, output, rows, self.hidden, self.padded_hidden)
    }

    fn pad(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
        columns: usize,
        padded_columns: usize,
    ) -> Result<()> {
        let elements = rows * padded_columns;
        Ok(self.pad_rows.launch(
            stream,
            launch(elements)?,
            (input, output, narrow(rows)?, narrow(columns)?, narrow(padded_columns)?),
        )?)
    }

    pub fn gate_up(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.gate_up.launch(
            stream,
            launch(self.assignments * self.padded_intermediate)?,
            (
                input,
                bias,
                selected,
                output,
                narrow(self.assignments)?,
                narrow(self.intermediate)?,
                narrow(self.padded_intermediate)?,
                self.limit,
            ),
        )?)
    }

    pub fn down_reduce(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.down_reduce.launch(
            stream,
            launch(self.tokens * self.hidden)?,
            (
                input,
                bias,
                selected,
                routing,
                output,
                narrow(self.tokens)?,
                narrow(self.top_k)?,
                narrow(self.hidden)?,
                narrow(self.padded_hidden)?,
            ),
        )?)
    }
}

fn launch(elements: usize) -> Result<LaunchConfig> {
    let block = 256usize;
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(block))?, 1, 1),
        block: (narrow(block)?, 1, 1),
        shared_memory_bytes: 0,
    })
}
