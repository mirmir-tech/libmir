//! GGML output scales (`kernels/ggml/scale.cu`): llama.cpp multiplies a
//! projection's products by its F32 `.scale` after the matmul, one factor
//! per matrix or per expert of a bank.

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::GgmlMmqPlacement;
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(
    ScaleKernel = "libmir_cuda_ggml_scale_bf16"(
        data: &mut DeviceBuffer<bf16>, scale: &DeviceBuffer<f32>, elements: u32,
    )
);
cuda_export!(
    ScaleExpertRowsKernel = "libmir_cuda_ggml_scale_expert_rows_bf16"(
        data: &mut DeviceBuffer<bf16>, scales: &DeviceBuffer<f32>, experts: &DeviceBuffer<u32>,
        rows: u32, columns: u32, ld: u32, stride: u32, offset: u32, expert_count: u32,
    )
);

cuda_export!(
    BiasKernel = "libmir_cuda_ggml_bias_bf16"(
        data: &mut DeviceBuffer<bf16>, bias: &DeviceBuffer<bf16>, rows: u32, columns: u32,
    )
);

/// The products of `rows` selection rows, `columns` of each at `placement`.
#[derive(Clone, Copy, Debug)]
pub struct GgmlScaledRows<'a> {
    /// The expert of every row.
    pub experts: &'a DeviceBuffer<u32>,
    pub rows: usize,
    pub columns: usize,
    pub placement: GgmlMmqPlacement,
}

/// Multiplies GGML projection products by their output scales.
#[derive(Clone, Debug)]
pub struct GgmlOutputScaling {
    matrix: TypedKernel<ScaleKernel>,
    rows: TypedKernel<ScaleExpertRowsKernel>,
    bias: TypedKernel<BiasKernel>,
}

impl GgmlOutputScaling {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/ggml/scale.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            matrix: module.kernel()?,
            rows: module.kernel()?,
            bias: module.kernel()?,
        })
    }

    /// The first `elements` products of one matrix times its one scale.
    pub fn matrix(
        &self,
        stream: &Stream,
        data: &mut DeviceBuffer<bf16>,
        scale: &DeviceBuffer<f32>,
        elements: usize,
    ) -> Result<()> {
        require("GGML output scale", 1, scale.len())?;
        if elements > data.len() {
            return Err(Error::InvalidExecutionPlan("GGML scaled products exceed the buffer"));
        }
        Ok(self
            .matrix
            .launch(stream, linear(elements)?, (data, scale, narrow(elements)?))?)
    }

    /// Adds `bias` to each of `rows` output rows.
    pub fn bias(
        &self,
        stream: &Stream,
        data: &mut DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>,
        rows: usize,
    ) -> Result<()> {
        let columns = bias.len();
        let elements = product(rows, columns)?;
        if elements > data.len() {
            return Err(Error::InvalidExecutionPlan("GGML biased products exceed the buffer"));
        }
        Ok(self.bias.launch(
            stream,
            linear(elements)?,
            (data, bias, narrow(rows)?, narrow(columns)?),
        )?)
    }

    /// Every selection row's products times its expert's scale of `scales`.
    pub fn expert_rows(
        &self,
        stream: &Stream,
        data: &mut DeviceBuffer<bf16>,
        scales: &DeviceBuffer<f32>,
        rows: GgmlScaledRows<'_>,
    ) -> Result<()> {
        let GgmlScaledRows { experts, rows: count, columns, placement } = rows;
        require("GGML scaled row experts", count, experts.len())?;
        let last = count
            .checked_sub(1)
            .and_then(|row| row.checked_mul(placement.ld))
            .and_then(|start| start.checked_add((columns.checked_sub(1)?) * placement.stride))
            .and_then(|end| end.checked_add(placement.offset));
        if last.is_none_or(|last| last >= data.len()) {
            return Err(Error::InvalidExecutionPlan("GGML scaled rows exceed the buffer"));
        }
        let elements = product(count, columns)?;
        Ok(self.rows.launch(
            stream,
            linear(elements)?,
            (
                data,
                scales,
                experts,
                narrow(count)?,
                narrow(columns)?,
                narrow(placement.ld)?,
                narrow(placement.stride)?,
                narrow(placement.offset)?,
                narrow(scales.len())?,
            ),
        )?)
    }
}

fn linear(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256).max(1))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
