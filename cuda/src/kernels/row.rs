use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    SelectRowKernel = "libmir_cuda_select_row_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        row: u32, rows: u32, columns: u32,
    )
);

#[derive(Clone, Debug)]
pub struct SelectRowBf16 {
    kernel: TypedKernel<SelectRowKernel>,
    columns: usize,
}

impl SelectRowBf16 {
    pub fn compile(compiler: &Compiler, columns: usize) -> Result<Self> {
        if columns == 0 {
            return Err(Error::InvalidDecoderKernel("row selection has no columns"));
        }
        let source = cuda_kernel_file!("../../kernels/select_row_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, columns })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        row: usize,
        rows: usize,
    ) -> Result<()> {
        if rows == 0 || row >= rows {
            return Err(Error::InvalidDecoderKernel("invalid selected row"));
        }
        require("row selection input", product(rows, self.columns)?, input.len())?;
        require("row selection output", self.columns, output.len())?;
        let threads = 256_usize;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.columns.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, output, narrow(row)?, narrow(rows)?, narrow(self.columns)?),
        )?)
    }
}
