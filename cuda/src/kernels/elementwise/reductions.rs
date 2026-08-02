use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{ElementwiseBf16, require};
use crate::{Error, Result};

impl ElementwiseBf16 {
    pub fn weighted_reduce(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weights: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
    ) -> Result<()> {
        self.weighted_reduce_batch(stream, input, weights, output, rows, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn weighted_reduce_batch(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weights: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let input_elements = self
            .elements
            .checked_mul(rows)
            .and_then(|elements| elements.checked_mul(tokens))
            .ok_or(Error::InvalidDecoderKernel("weighted reduction overflow"))?;
        let weight_elements = rows
            .checked_mul(tokens)
            .ok_or(Error::InvalidDecoderKernel("weighted reduction overflow"))?;
        let output_elements = self
            .elements
            .checked_mul(tokens)
            .ok_or(Error::InvalidDecoderKernel("weighted reduction overflow"))?;
        require("weighted reduction input", input_elements, input.len())?;
        require("weighted reduction weights", weight_elements, weights.len())?;
        require("weighted reduction output", output_elements, output.len())?;
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("weighted reduction batch is empty"));
        }
        let threads = 256_usize;
        Ok(self.weighted_reduce.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(output_elements.div_ceil(threads))?, 1, 1),
                block: (u32::try_from(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weights,
                output,
                u32::try_from(rows)?,
                self.count()?,
                u32::try_from(tokens)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn weighted_reduce_bucketed(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weights: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let assignments = rows
            .checked_mul(tokens)
            .ok_or(Error::InvalidDecoderKernel("bucketed reduction overflow"))?;
        let input_elements = self
            .elements
            .checked_mul(assignments)
            .ok_or(Error::InvalidDecoderKernel("bucketed reduction overflow"))?;
        let output_elements = self
            .elements
            .checked_mul(tokens)
            .ok_or(Error::InvalidDecoderKernel("bucketed reduction overflow"))?;
        require("bucketed reduction input", input_elements, input.len())?;
        require("bucketed reduction weights", assignments, weights.len())?;
        require("bucketed reduction positions", assignments, positions.len())?;
        require("bucketed reduction output", output_elements, output.len())?;
        let threads = 256_usize;
        Ok(self.weighted_reduce_bucketed.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(output_elements.div_ceil(threads))?, 1, 1),
                block: (u32::try_from(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weights,
                positions,
                output,
                u32::try_from(rows)?,
                self.count()?,
                u32::try_from(tokens)?,
            ),
        )?)
    }
}
