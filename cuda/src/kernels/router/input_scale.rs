//! The input scaling of a router that weighs its one selected expert's input
//! (Llama 4).

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    RouterInputScaleKernel = "libmir_cuda_router_input_scale_bf16"(
        input: &DeviceBuffer<bf16>, weights: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
    )
);

#[derive(Clone, Debug)]
pub struct RouterInputScale {
    kernel: TypedKernel<RouterInputScaleKernel>,
    columns: usize,
}

impl RouterInputScale {
    pub fn compile(compiler: &Compiler, columns: usize) -> Result<Self> {
        if columns == 0 {
            return Err(Error::InvalidRouter("router input scaling has no columns"));
        }
        let source = cuda_kernel_file!("../../../kernels/router_input_scale_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, columns })
    }

    /// Writes each `input` row times its one routing weight into `output`
    /// and leaves unit `weights` for the experts.
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weights: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let rows = weights.len();
        require("router input", product(rows, self.columns)?, input.len())?;
        require("scaled router input", product(rows, self.columns)?, output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(rows)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, weights, output, narrow(rows)?, narrow(self.columns)?),
        )?)
    }
}
