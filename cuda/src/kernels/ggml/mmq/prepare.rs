//! Quantizing a grouped projection's input rows and laying out its tiles.

use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16, cuda_export};

use super::{
    super::super::geometry::{narrow, product, require},
    GgmlActivations, GgmlGroupedMmq, GgmlMmqGroups,
};
use crate::{Error, Result};

cuda_export!(pub(super) TilesKernel = "libmir_cuda_ggml_mmq_tiles"(
    offsets: &DeviceBuffer<u32>, counts: &DeviceBuffer<u32>,
    tile_experts: &mut DeviceBuffer<u32>, tile_rows: &mut DeviceBuffer<u32>, tile_count: &mut DeviceBuffer<u32>, experts: u32,
    max_tiles: u32,
));

cuda_export!(pub(super) GatherKernel = "libmir_cuda_ggml_mmq_gather_q8"(
    input: &DeviceBuffer<bf16>, assignments: &DeviceBuffer<u32>,
    quantized: &mut DeviceBuffer<u32>, scales: &mut DeviceBuffer<f32>, rows: u32, columns: u32,
    top_k: u32,
));

/// The rows a grouped projection quantizes: `rows` in expert order, or the
/// tokens behind each expert-order selection (`assignments[i] / top_k`).
#[derive(Clone, Copy)]
pub enum GgmlMmqInput<'a> {
    Rows(&'a DeviceBuffer<bf16>),
    Gathered {
        tokens: &'a DeviceBuffer<bf16>,
        assignments: &'a DeviceBuffer<u32>,
        top_k: usize,
    },
}

impl GgmlGroupedMmq {
    /// Quantizes `count` rows of `input` and lays out the tiles of the
    /// expert groups.
    pub fn prepare(
        &mut self,
        stream: &Stream,
        input: GgmlMmqInput<'_>,
        count: usize,
        groups: GgmlMmqGroups<'_>,
        activations: &mut GgmlActivations,
    ) -> Result<()> {
        let subs = self.require(count, activations)?;
        require("GGML grouped offsets", self.experts, groups.offsets.len())?;
        require("GGML grouped counts", self.experts, groups.counts.len())?;
        let (quantized, scales) = activations.buffers_mut();
        let config = LaunchConfig {
            grid: (narrow(subs.div_ceil(128))?, 1, 1),
            block: (128, 1, 1),
            shared_memory_bytes: 0,
        };
        match input {
            GgmlMmqInput::Rows(rows) => {
                require("GGML grouped input", product(count, self.columns)?, rows.len())?;
                self.quantize.launch(
                    stream,
                    config,
                    (rows, quantized, scales, narrow(count)?, narrow(self.columns)?),
                )?;
            },
            GgmlMmqInput::Gathered { tokens, assignments, top_k } => {
                require("GGML grouped assignments", count, assignments.len())?;
                if top_k == 0 || !tokens.len().is_multiple_of(self.columns) {
                    return Err(Error::InvalidExecutionPlan("invalid GGML gathered input"));
                }
                self.gather.launch(
                    stream,
                    config,
                    (
                        tokens,
                        assignments,
                        quantized,
                        scales,
                        narrow(count)?,
                        narrow(self.columns)?,
                        narrow(top_k)?,
                    ),
                )?;
            },
        }
        self.tiles.launch(
            stream,
            LaunchConfig {
                grid: (1, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                groups.offsets,
                groups.counts,
                &mut self.tile_experts,
                &mut self.tile_rows,
                &mut self.tile_count,
                narrow(self.experts)?,
                narrow(self.max_tiles)?,
            ),
        )?;
        Ok(())
    }

    pub(super) fn require(&self, count: usize, activations: &GgmlActivations) -> Result<usize> {
        if count > self.max_rows {
            return Err(Error::InvalidExecutionPlan("GGML grouped GEMM has too many rows"));
        }
        let subs = product(count, self.columns / 32)?;
        activations.require(subs)?;
        Ok(subs)
    }
}
