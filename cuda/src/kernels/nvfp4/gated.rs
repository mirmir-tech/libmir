use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::scale_elements;
use crate::{
    Error, Result,
    kernels::{GatedActivation, geometry::require},
};

cuda_export!(
    GatedKernel = "libmir_cuda_gated_nvfp4"(
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        global_scale: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        rows: u32,
        columns: u32,
        activation: u32,
    )
);

/// Gated activation rounded to BF16 and packed directly as NVFP4.
#[derive(Clone, Debug)]
pub struct NvFp4Gated {
    kernel: TypedKernel<GatedKernel>,
    rows: usize,
    columns: usize,
}

impl NvFp4Gated {
    pub fn compile(compiler: &Compiler, rows: usize, columns: usize) -> Result<Self> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(64) {
            return Err(Error::InvalidNvFp4("invalid gated NVFP4 geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/gated_nvfp4.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, rows, columns })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        global_scale: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        activation: GatedActivation,
    ) -> Result<()> {
        let elements = self
            .rows
            .checked_mul(self.columns)
            .ok_or(Error::InvalidNvFp4("gated NVFP4 size overflow"))?;
        require("gated NVFP4 gate", elements, gate.len())?;
        require("gated NVFP4 up", elements, up.len())?;
        require("gated NVFP4 global scale", 1, global_scale.len())?;
        require("gated NVFP4 packed output", elements / 2, packed.len())?;
        require("gated NVFP4 scales", scale_elements(self.rows, self.columns)?, scales.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(elements / 16)?, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                gate,
                up,
                global_scale,
                packed,
                scales,
                u32::try_from(self.rows)?,
                u32::try_from(self.columns)?,
                activation.code(),
            ),
        )?)
    }
}
