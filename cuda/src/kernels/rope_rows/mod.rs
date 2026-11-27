use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

#[cfg(test)]
mod tests;

cuda_export!(RopeRowsKernel = "libmir_cuda_rope_rows_u8"(
    source: &DeviceBuffer<u8>, output: &mut DeviceBuffer<u8>, rows: u32, row_bytes: u32,
    head_dim: u32, rotary_dim: u32,
));

/// Query and key rows from llama.cpp's pairwise-interleaved order back to
/// Hugging Face's `rotate_half` halves, per head of `head_dim` rows.
#[derive(Clone, Debug)]
pub struct RopeRows {
    kernel: TypedKernel<RopeRowsKernel>,
}

impl RopeRows {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/rope_rows_u8.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self { kernel: module.kernel()? })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<u8>,
        rows: usize,
        (head_dim, rotary_dim): (usize, usize),
    ) -> Result<()> {
        // No rows pass the head check but leave no row width.
        if rows == 0
            || head_dim == 0
            || rotary_dim == 0
            || !rotary_dim.is_multiple_of(2)
            || rotary_dim > head_dim
            || !rows.is_multiple_of(head_dim)
        {
            return Err(Error::InvalidDecoderKernel("rotary rows need whole even heads"));
        }
        let row_bytes = source.len() / rows;
        if row_bytes == 0 {
            return Err(Error::InvalidDecoderKernel("rotary rows hold no bytes"));
        }
        require("rotary rows", product(rows, row_bytes)?, source.len())?;
        require("restored rotary rows", source.len(), output.len())?;
        self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(source.len().div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                source,
                output,
                narrow(rows)?,
                narrow(row_bytes)?,
                narrow(head_dim)?,
                narrow(rotary_dim)?,
            ),
        )?;
        Ok(())
    }
}
