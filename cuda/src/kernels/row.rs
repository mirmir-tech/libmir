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

cuda_export!(
    CopyRowsKernel = "libmir_cuda_copy_rows_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        input_start: u32, output_start: u32, rows: u32, columns: u32,
    )
);

cuda_export!(
    GatherRowsKernel = "libmir_cuda_gather_rows_bf16"(
        input: &DeviceBuffer<bf16>, indices: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>, input_rows: u32,
        output_rows: u32, columns: u32,
    )
);

#[derive(Clone, Debug)]
pub struct SelectRowBf16 {
    kernel: TypedKernel<SelectRowKernel>,
    columns: usize,
}

#[derive(Clone, Debug)]
pub struct CopyRowsBf16 {
    kernel: TypedKernel<CopyRowsKernel>,
    columns: usize,
}

#[derive(Clone, Debug)]
pub struct GatherRowsBf16 {
    kernel: TypedKernel<GatherRowsKernel>,
    columns: usize,
}

impl CopyRowsBf16 {
    pub fn compile(compiler: &Compiler, columns: usize) -> Result<Self> {
        if columns == 0 {
            return Err(Error::InvalidDecoderKernel("row copy has no columns"));
        }
        let source = cuda_kernel_file!("../../kernels/select_row_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, columns })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        input_start: usize,
        output_start: usize,
        rows: usize,
    ) -> Result<()> {
        let input_end = input_start
            .checked_add(rows)
            .ok_or(Error::InvalidDecoderKernel("row copy input overflow"))?;
        let output_end = output_start
            .checked_add(rows)
            .ok_or(Error::InvalidDecoderKernel("row copy output overflow"))?;
        require("row copy input", product(input_end, self.columns)?, input.len())?;
        require("row copy output", product(output_end, self.columns)?, output.len())?;
        let elements = product(rows, self.columns)?;
        let threads = 256_usize;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                output,
                narrow(input_start)?,
                narrow(output_start)?,
                narrow(rows)?,
                narrow(self.columns)?,
            ),
        )?)
    }
}

impl GatherRowsBf16 {
    pub fn compile(compiler: &Compiler, columns: usize) -> Result<Self> {
        if columns == 0 {
            return Err(Error::InvalidDecoderKernel("row gather has no columns"));
        }
        let source = cuda_kernel_file!("../../kernels/select_row_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, columns })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        indices: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
        input_rows: usize,
    ) -> Result<()> {
        let output_rows = indices.len();
        require("row gather input", product(input_rows, self.columns)?, input.len())?;
        require("row gather output", product(output_rows, self.columns)?, output.len())?;
        let elements = product(output_rows, self.columns)?;
        let threads = 256_usize;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                indices,
                output,
                narrow(input_rows)?,
                narrow(output_rows)?,
                narrow(self.columns)?,
            ),
        )?)
    }
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

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::{CudaBackend, CudaConfig};

    #[test]
    fn gathers_non_contiguous_rows_in_output_order() -> Result<()> {
        let backend = CudaBackend::new(CudaConfig::default())?;
        let (context, stream, pool, compiler) = backend.test_resources();
        let values = (0_u8..12).map(|value| bf16::from_f32(f32::from(value))).collect::<Vec<_>>();
        let mut input_host = context.allocate_pinned(values.len())?;
        input_host.copy_from_slice(&values)?;
        let mut input = pool.allocate(stream, values.len())?;
        stream.copy_to_device(&mut input_host, &mut input)?;

        let mut index_host = context.allocate_pinned(2)?;
        index_host.copy_from_slice(&[3, 1])?;
        let mut indices = pool.allocate(stream, 2)?;
        stream.copy_to_device(&mut index_host, &mut indices)?;
        let mut output = pool.allocate(stream, 6)?;
        GatherRowsBf16::compile(compiler, 3)?.execute(stream, &input, &indices, &mut output, 4)?;

        let mut actual = context.allocate_pinned(6)?;
        stream.copy_to_host(&output, &mut actual)?;
        assert_eq!(
            actual.to_vec()?.into_iter().map(bf16::to_f32).collect::<Vec<_>>(),
            [9.0, 10.0, 11.0, 3.0, 4.0, 5.0],
        );
        Ok(())
    }
}
