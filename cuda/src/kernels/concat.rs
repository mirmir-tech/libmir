use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    ConcatColumnsKernel = "libmir_cuda_concat_columns_bf16"(
        left: &DeviceBuffer<bf16>, right: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
    )
);

/// Joins two `[rows, columns]` matrices side by side into `[rows, 2 ·
/// columns]`.
#[derive(Clone, Debug)]
pub struct ConcatColumnsBf16 {
    kernel: TypedKernel<ConcatColumnsKernel>,
    columns: usize,
}

impl ConcatColumnsBf16 {
    pub fn compile(compiler: &Compiler, columns: usize) -> Result<Self> {
        if columns == 0 {
            return Err(Error::InvalidDecoderKernel("column concatenation has no columns"));
        }
        let source = cuda_kernel_file!("../../kernels/concat_columns_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, columns })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<bf16>,
        right: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
    ) -> Result<()> {
        let half = product(rows, self.columns)?;
        require("concatenation left input", half, left.len())?;
        require("concatenation right input", half, right.len())?;
        let elements = product(half, 2)?;
        require("concatenation output", elements, output.len())?;
        let threads = 256_usize;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (left, right, output, narrow(rows)?, narrow(self.columns)?),
        )?)
    }
}
