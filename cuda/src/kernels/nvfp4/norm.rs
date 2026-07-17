use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::scale_elements;
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(
    InverseKernel = "libmir_cuda_rms_inverse_bf16"(
        input: &DeviceBuffer<bf16>,
        inverse: &mut DeviceBuffer<f32>,
        rows: u32,
        columns: u32,
        epsilon: f32,
    )
);

cuda_export!(
    QuantizeKernel = "libmir_cuda_rms_norm_nvfp4"(
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        inverse: &DeviceBuffer<f32>,
        global_scale: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        rows: u32,
        columns: u32,
    )
);

/// RMS reduction followed by parallel normalized NVFP4 packing.
#[derive(Clone, Debug)]
pub struct NvFp4RmsNorm {
    inverse: TypedKernel<InverseKernel>,
    quantize: TypedKernel<QuantizeKernel>,
    rows: usize,
    columns: usize,
}

impl NvFp4RmsNorm {
    pub fn compile(compiler: &Compiler, rows: usize, columns: usize) -> Result<Self> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(64) {
            return Err(Error::InvalidNvFp4("invalid fused RMSNorm geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/rms_norm_nvfp4.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            inverse: module.kernel()?,
            quantize: module.kernel()?,
            rows,
            columns,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        global_scale: &DeviceBuffer<f32>,
        inverse: &mut DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        epsilon: f32,
    ) -> Result<()> {
        self.validate(input, weight, global_scale, inverse, packed, scales, epsilon)?;
        self.inverse.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.rows)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, &mut *inverse, narrow(self.rows)?, narrow(self.columns)?, epsilon),
        )?;
        Ok(self.quantize.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.rows * self.columns / 16)?, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weight,
                &*inverse,
                global_scale,
                packed,
                scales,
                narrow(self.rows)?,
                narrow(self.columns)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        global_scale: &DeviceBuffer<f32>,
        inverse: &DeviceBuffer<f32>,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        epsilon: f32,
    ) -> Result<()> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(Error::InvalidNvFp4("invalid fused RMSNorm epsilon"));
        }
        let elements = product(self.rows, self.columns)?;
        require("fused RMSNorm input", elements, input.len())?;
        require("fused RMSNorm weight", self.columns, weight.len())?;
        require("fused RMSNorm global scale", 1, global_scale.len())?;
        require("fused RMSNorm inverse", self.rows, inverse.len())?;
        require("fused RMSNorm packed output", elements / 2, packed.len())?;
        require("fused RMSNorm scales", scale_elements(self.rows, self.columns)?, scales.len())
    }
}
