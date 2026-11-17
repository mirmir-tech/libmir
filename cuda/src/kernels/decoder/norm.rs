//! Row RMS normalization, with and without a weight.

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    RmsNormKernel = "libmir_cuda_rms_norm_bf16"(
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: u32,
        columns: u32,
        epsilon: f32,
    )
);

cuda_export!(
    RmsNormUnitKernel = "libmir_cuda_rms_norm_unit_bf16"(
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: u32,
        columns: u32,
        epsilon: f32,
    )
);

#[derive(Clone, Debug)]
pub struct RmsNorm {
    kernel: TypedKernel<RmsNormKernel>,
    rows: usize,
    columns: usize,
    epsilon: f32,
}

#[derive(Clone, Debug)]
pub struct RmsNormUnit {
    kernel: TypedKernel<RmsNormUnitKernel>,
    rows: usize,
    columns: usize,
    epsilon: f32,
}

impl RmsNorm {
    pub fn compile(compiler: &Compiler, rows: usize, columns: usize, epsilon: f32) -> Result<Self> {
        if rows == 0 || columns == 0 || !epsilon.is_finite() || epsilon < 0.0 {
            return Err(Error::InvalidDecoderKernel("invalid RMSNorm geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/rms_norm_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            rows,
            columns,
            epsilon,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(self.rows, self.columns)?;
        require("RMSNorm input", elements, input.len())?;
        require("RMSNorm weight", self.columns, weight.len())?;
        require("RMSNorm output", elements, output.len())?;
        let config = LaunchConfig {
            grid: (narrow(self.rows)?, 1, 1),
            block: (rms_norm_threads(self.columns)?, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.kernel.launch(
            stream,
            config,
            (input, weight, output, narrow(self.rows)?, narrow(self.columns)?, self.epsilon),
        )?)
    }
}

impl RmsNormUnit {
    pub fn compile(compiler: &Compiler, rows: usize, columns: usize, epsilon: f32) -> Result<Self> {
        if rows == 0 || columns == 0 || !epsilon.is_finite() || epsilon < 0.0 {
            return Err(Error::InvalidDecoderKernel("invalid unit RMSNorm geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/rms_norm_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            rows,
            columns,
            epsilon,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(self.rows, self.columns)?;
        require("unit RMSNorm input", elements, input.len())?;
        require("unit RMSNorm output", elements, output.len())?;
        let config = LaunchConfig {
            grid: (narrow(self.rows)?, 1, 1),
            block: (rms_norm_threads(self.columns)?, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.kernel.launch(
            stream,
            config,
            (input, output, narrow(self.rows)?, narrow(self.columns)?, self.epsilon),
        )?)
    }
}

/// Threads per `RMSNorm` row: one per 16-byte vector of the row, in whole
/// warps, at most 1024 (`rms_norm_bf16.cu` caches four vectors per thread).
fn rms_norm_threads(columns: usize) -> Result<u32> {
    narrow(columns.div_ceil(8).next_multiple_of(32).clamp(32, 1024))
}
