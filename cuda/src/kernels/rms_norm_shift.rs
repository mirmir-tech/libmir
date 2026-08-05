use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    RmsNormShiftKernel = "libmir_cuda_rms_norm_shift_bf16"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
        epsilon: f32, weight_shift: f32,
    )
);
cuda_export!(
    ResidualRmsNormShiftKernel = "libmir_cuda_residual_rms_norm_shift_bf16"(
        input: &DeviceBuffer<bf16>, update: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>, residual: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
        epsilon: f32, weight_shift: f32,
    )
);

#[derive(Clone, Debug)]
pub struct ShiftedRmsNorm {
    kernel: TypedKernel<RmsNormShiftKernel>,
    residual_kernel: TypedKernel<ResidualRmsNormShiftKernel>,
    rows: usize,
    columns: usize,
    epsilon: f32,
    weight_shift: f32,
}

impl ShiftedRmsNorm {
    pub fn compile(
        compiler: &Compiler,
        rows: usize,
        columns: usize,
        epsilon: f32,
        weight_shift: f32,
    ) -> Result<Self> {
        if rows == 0
            || columns == 0
            || !epsilon.is_finite()
            || epsilon < 0.0
            || !weight_shift.is_finite()
        {
            return Err(Error::InvalidDecoderKernel("invalid shifted RMSNorm geometry"));
        }
        let source = cuda_kernel_file!("../../kernels/rms_norm_shift_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            residual_kernel: module.kernel()?,
            rows,
            columns,
            epsilon,
            weight_shift,
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
        require("shifted RMSNorm input", elements, input.len())?;
        require("shifted RMSNorm weight", self.columns, weight.len())?;
        require("shifted RMSNorm output", elements, output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.rows)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weight,
                output,
                narrow(self.rows)?,
                narrow(self.columns)?,
                self.epsilon,
                self.weight_shift,
            ),
        )?)
    }

    pub fn execute_residual(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        update: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        residual: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(self.rows, self.columns)?;
        require("residual RMSNorm input", elements, input.len())?;
        require("residual RMSNorm update", elements, update.len())?;
        require("residual RMSNorm weight", self.columns, weight.len())?;
        require("residual RMSNorm residual", elements, residual.len())?;
        require("residual RMSNorm output", elements, output.len())?;
        Ok(self.residual_kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.rows)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                update,
                weight,
                residual,
                output,
                narrow(self.rows)?,
                narrow(self.columns)?,
                self.epsilon,
                self.weight_shift,
            ),
        )?)
    }
}
