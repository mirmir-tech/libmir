//! GGML MXFP4 expert banks repacked to the OCP MXFP4 layout on the device
//! (`kernels/ggml/repack.cu`).

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::Result;

cuda_export!(MxFp4ToOcp = "libmir_cuda_ggml_mxfp4_to_ocp"(
    source: &DeviceBuffer<u8>, blocks: &mut DeviceBuffer<u8>, scales: &mut DeviceBuffer<u8>,
    rows: u32, blocks_per_row: u32, stride: u32, offset: u32,
));

cuda_export!(InterleaveBf16 = "libmir_cuda_ggml_interleave_bf16"(
    source: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, count: u32, stride: u32,
    offset: u32,
));

/// Rows land at `row * stride + offset`: stride 2 interleaves a gate and an
/// up bank into one fused gate/up bank.
#[derive(Clone, Copy, Debug)]
pub struct GgmlRowPlacement {
    pub stride: usize,
    pub offset: usize,
}

#[derive(Clone, Debug)]
pub struct GgmlMxFp4Repack {
    repack: TypedKernel<MxFp4ToOcp>,
    interleave: TypedKernel<InterleaveBf16>,
}

impl GgmlMxFp4Repack {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/ggml/repack.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self {
            repack: module.kernel()?,
            interleave: module.kernel()?,
        })
    }

    /// `rows` rows of `columns` values from GGML `source` into OCP `blocks`
    /// (16 bytes per 32 values) and `scales` (one byte per 32 values).
    #[allow(clippy::too_many_arguments)]
    pub fn blocks(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<u8>,
        blocks: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        rows: usize,
        columns: usize,
        placement: GgmlRowPlacement,
    ) -> Result<()> {
        let per_row = columns / 32;
        let total = product(rows, per_row)?;
        let placed = product(product(rows, placement.stride)?, per_row)?;
        require("GGML MXFP4 source", product(total, 17)?, source.len())?;
        require("OCP MXFP4 blocks", product(placed, 16)?, blocks.len())?;
        require("OCP MXFP4 scales", placed, scales.len())?;
        self.repack.launch(
            stream,
            launch(total)?,
            (
                source,
                blocks,
                scales,
                narrow(rows)?,
                narrow(per_row)?,
                narrow(placement.stride)?,
                narrow(placement.offset)?,
            ),
        )?;
        Ok(())
    }

    pub fn bias(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        placement: GgmlRowPlacement,
    ) -> Result<()> {
        require("interleaved bias", product(source.len(), placement.stride)?, output.len())?;
        self.interleave.launch(
            stream,
            launch(source.len())?,
            (
                source,
                output,
                narrow(source.len())?,
                narrow(placement.stride)?,
                narrow(placement.offset)?,
            ),
        )?;
        Ok(())
    }
}

fn launch(threads: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(threads.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
