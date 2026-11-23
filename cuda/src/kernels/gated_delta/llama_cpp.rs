//! llama.cpp's GGUF Gated `DeltaNet` layout
//! (`kernels/gated_delta_llama_cpp_bf16.cu`): the decay log from the stored
//! `-exp(A_log)`, value-head rows reordered to grouped order, and the
//! attention output tiled back for the stored output projection.

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(
    DecayLogKernel = "libmir_cuda_gated_delta_decay_log_bf16"(
        values: &mut DeviceBuffer<bf16>, count: u32,
    )
);
cuda_export!(
    GatherRowsKernel = "libmir_cuda_gated_delta_gather_rows"(
        input: &DeviceBuffer<u8>, output: &mut DeviceBuffer<u8>, rows: &DeviceBuffer<u32>,
        count: u32, row_bytes: u32,
    )
);
cuda_export!(
    TileValuesKernel = "libmir_cuda_gated_delta_tile_values_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, tokens: u32,
        key_heads: u32, per_key: u32, head_dim: u32,
    )
);

/// Value heads stored tiled across key heads: `key_heads * per_key` heads of
/// `head_dim` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedDeltaTiledValues {
    pub key_heads: usize,
    pub per_key: usize,
    pub head_dim: usize,
}

impl GatedDeltaTiledValues {
    /// Stored rows of every grouped row: `offset` leading rows in place, then
    /// `width` rows per value head, grouped head `k * per_key + j` read from
    /// stored head `j * key_heads + k`.
    pub fn rows(self, offset: usize, width: usize) -> Result<Vec<u32>> {
        let heads = product(self.key_heads, self.per_key)?;
        let mut rows =
            (0..offset).map(u32::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
        for grouped in 0..heads {
            let stored = (grouped % self.per_key) * self.key_heads + grouped / self.per_key;
            for row in 0..width {
                rows.push(u32::try_from(offset + stored * width + row)?);
            }
        }
        Ok(rows)
    }
}

#[derive(Clone, Debug)]
pub struct GatedDeltaLlamaCpp {
    decay_log: TypedKernel<DecayLogKernel>,
    gather: TypedKernel<GatherRowsKernel>,
    tile: TypedKernel<TileValuesKernel>,
}

impl GatedDeltaLlamaCpp {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/gated_delta_llama_cpp_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            decay_log: module.kernel()?,
            gather: module.kernel()?,
            tile: module.kernel()?,
        })
    }

    /// `A_log = ln(-ssm_a)` in place.
    pub fn decay_log(&self, stream: &Stream, values: &mut DeviceBuffer<bf16>) -> Result<()> {
        let count = values.len();
        Ok(self.decay_log.launch(stream, linear(count)?, (values, narrow(count)?))?)
    }

    /// `output` row `r` is `input` row `rows[r]`, `row_bytes` each.
    pub fn gather_rows(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<u8>,
        rows: &DeviceBuffer<u32>,
        row_bytes: usize,
    ) -> Result<()> {
        let bytes = product(rows.len(), row_bytes)?;
        require("gathered Gated Delta rows", bytes, output.len())?;
        if bytes > input.len() {
            return Err(Error::InvalidExecutionPlan("gathered Gated Delta rows exceed the input"));
        }
        let arguments = (input, output, rows, narrow(rows.len())?, narrow(row_bytes)?);
        Ok(self.gather.launch(stream, linear(bytes)?, arguments)?)
    }

    /// The `tokens` grouped attention outputs in tiled value-head order.
    pub fn tile_values(
        &self,
        stream: &Stream,
        tiled: GatedDeltaTiledValues,
        (input, output): (&DeviceBuffer<bf16>, &mut DeviceBuffer<bf16>),
        tokens: usize,
    ) -> Result<()> {
        let width = product(product(tiled.key_heads, tiled.per_key)?, tiled.head_dim)?;
        let elements = product(tokens, width)?;
        if elements > input.len() || elements > output.len() {
            return Err(Error::InvalidExecutionPlan("tiled Gated Delta values exceed the buffer"));
        }
        let arguments = (
            input,
            output,
            narrow(tokens)?,
            narrow(tiled.key_heads)?,
            narrow(tiled.per_key)?,
            narrow(tiled.head_dim)?,
        );
        Ok(self.tile.launch(stream, linear(elements)?, arguments)?)
    }
}

fn linear(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256).max(1))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
