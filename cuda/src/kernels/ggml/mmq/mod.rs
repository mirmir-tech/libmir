//! Grouped GGML GEMM over stacked expert banks (`kernels/ggml/mmq.cu`):
//! input rows grouped by expert, quantized to int8 like decode activations,
//! times the expert's GGML rows on int8 tensor cores. Every type of a
//! checkpoint gets its kernel at compile; the bank's type picks one at
//! execute.

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, MemoryPool, Module, Stream, TypedKernel,
    bf16, cuda_kernel_files,
};

use super::{
    super::geometry::{narrow, product, require},
    GgmlActivations, GgmlKernelType,
    linear::QuantizeQ8,
};
use crate::{Error, Result};

mod combine;
mod kernels;
mod prepare;

pub use combine::GgmlRouteCombine;
use kernels::{MmqArguments, MmqKernel};
pub use prepare::GgmlMmqInput;
use prepare::{GatherKernel, TilesKernel};

/// The module of `kernels/ggml/mmq.cu` with the block decoders and the
/// activation quantizer.
fn module(compiler: &Compiler) -> Result<Module> {
    Ok(compiler.compile(
        cuda_kernel_files!(
            "ggml_mmq.cu";
            "../../../../kernels/ggml/blocks.cuh",
            "../../../../kernels/ggml/legacy.cuh",
            "../../../../kernels/ggml/k_quants.cuh",
            "../../../../kernels/ggml/iq4.cuh",
            "../../../../kernels/ggml/grids/iq2.cuh",
            "../../../../kernels/ggml/grids/iq13.cuh",
            "../../../../kernels/ggml/iq.cuh",
            "../../../../kernels/ggml/dot.cuh",
            "../../../../kernels/ggml/staged.cu",
            "../../../../kernels/ggml/vector.cu",
            "../../../../kernels/ggml/mma.cuh",
            "../../../../kernels/ggml/mmq.cu",
            "../../../../kernels/ggml/route.cu",
        ),
        &CompileOptions::default(),
    )?)
}

/// Rows and columns of each output tile (`kTileRows`, `kTileColumns`).
const TILE: usize = 64;

/// Where a projection writes: column `c` of row `r` lands at
/// `r * ld + c * stride + offset`, which interleaves gate and up outputs.
#[derive(Clone, Copy, Debug)]
pub struct GgmlMmqPlacement {
    pub ld: usize,
    pub stride: usize,
    pub offset: usize,
}

/// The expert groups of the input rows: rows `offsets[e]..offsets[e] +
/// counts[e]` belong to expert `e`.
#[derive(Clone, Copy)]
pub struct GgmlMmqGroups<'a> {
    pub offsets: &'a DeviceBuffer<u32>,
    pub counts: &'a DeviceBuffer<u32>,
}

/// Grouped projections of up to `max_rows` input rows over banks of
/// `experts` matrices of `rows` by `columns`.
#[derive(Debug)]
pub struct GgmlGroupedMmq {
    quantize: TypedKernel<QuantizeQ8>,
    gather: TypedKernel<GatherKernel>,
    tiles: TypedKernel<TilesKernel>,
    kernels: Vec<(GgmlKernelType, MmqKernel)>,
    experts: usize,
    rows: usize,
    columns: usize,
    max_rows: usize,
    max_tiles: usize,
    tile_experts: DeviceBuffer<u32>,
    tile_rows: DeviceBuffer<u32>,
    tile_count: DeviceBuffer<u32>,
}

impl GgmlGroupedMmq {
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        compiler: &Compiler,
        pool: &MemoryPool,
        stream: &Stream,
        kinds: &[GgmlKernelType],
        experts: usize,
        rows: usize,
        columns: usize,
        max_rows: usize,
    ) -> Result<Self> {
        if experts == 0 || rows == 0 || max_rows == 0 || !columns.is_multiple_of(32) {
            return Err(Error::InvalidDecoderKernel("invalid GGML grouped GEMM geometry"));
        }
        let module = module(compiler)?;
        let kernels = kinds
            .iter()
            .map(|&kind| Ok((kind, MmqKernel::load(&module, kind)?)))
            .collect::<Result<Vec<_>>>()?;
        // Each expert's rows end in at most one partial tile.
        let max_tiles = max_rows.div_ceil(TILE) + experts;
        Ok(Self {
            quantize: module.kernel()?,
            gather: module.kernel()?,
            tiles: module.kernel()?,
            kernels,
            experts,
            rows,
            columns,
            max_rows,
            max_tiles,
            tile_experts: pool.allocate(stream, max_tiles)?,
            tile_rows: pool.allocate(stream, max_tiles)?,
            tile_count: pool.allocate(stream, 1)?,
        })
    }

    /// Activations of `max_rows` input rows for [`Self::quantize`].
    pub fn activations(&self, pool: &MemoryPool, stream: &Stream) -> Result<GgmlActivations> {
        GgmlActivations::new(pool, stream, self.max_rows, self.columns)
    }

    /// The projection of the rows [`Self::prepare`] quantized through one
    /// bank of GGML blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn multiply(
        &self,
        stream: &Stream,
        activations: &GgmlActivations,
        groups: GgmlMmqGroups<'_>,
        kind: GgmlKernelType,
        weight: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        placement: GgmlMmqPlacement,
    ) -> Result<()> {
        let kernel = self
            .kernels
            .iter()
            .find(|(candidate, _)| *candidate == kind)
            .map(|(_, kernel)| kernel)
            .ok_or(Error::InvalidExecutionPlan("GGML grouped GEMM has no kernel for the bank"))?;
        require(
            "GGML grouped bank",
            product(self.experts, kind.bytes(self.rows, self.columns)?)?,
            weight.len(),
        )?;
        let last = (self.rows - 1) * placement.stride + placement.offset;
        if last >= placement.ld {
            return Err(Error::InvalidExecutionPlan("GGML grouped output placement overflows"));
        }
        require("GGML grouped output", product(self.max_rows, placement.ld)?, output.len())?;
        let (quantized, scales) = activations.buffers();
        kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.rows.div_ceil(TILE))?, narrow(self.max_tiles)?, 1),
                block: (128, 1, 1),
                shared_memory_bytes: 0,
            },
            MmqArguments {
                quantized,
                scales,
                offsets: groups.offsets,
                counts: groups.counts,
                tile_experts: &self.tile_experts,
                tile_rows: &self.tile_rows,
                tile_count: &self.tile_count,
                weight,
                output,
                rows: narrow(self.rows)?,
                columns: narrow(self.columns)?,
                ld: narrow(placement.ld)?,
                stride: narrow(placement.stride)?,
                offset: narrow(placement.offset)?,
            },
        )
    }
}
