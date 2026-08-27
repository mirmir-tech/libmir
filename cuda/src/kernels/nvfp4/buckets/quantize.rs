use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{BucketQuantize, NvFp4BucketPreparation};
use crate::{
    Error, Result,
    kernels::{
        geometry::{narrow, product, require},
        scale_elements,
    },
};

impl NvFp4BucketPreparation {
    #[allow(clippy::too_many_arguments)]
    pub fn quantize(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        order: &DeviceBuffer<u32>,
        offsets: &DeviceBuffer<u32>,
        scale_offsets: &DeviceBuffer<u32>,
        globals: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        geometry: BucketQuantize,
    ) -> Result<()> {
        geometry
            .validate(input, selected, order, offsets, scale_offsets, globals, packed, scales)?;
        let launch = launch(geometry)?;
        Ok(self.quantize.launch(
            stream,
            launch,
            (
                input,
                selected,
                order,
                offsets,
                scale_offsets,
                globals,
                packed,
                scales,
                narrow(geometry.assignments)?,
                narrow(geometry.selected)?,
                narrow(geometry.input_rows)?,
                narrow(geometry.columns)?,
                u32::from(geometry.ranked),
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn quantize_pair(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        order: &DeviceBuffer<u32>,
        offsets: &DeviceBuffer<u32>,
        scale_offsets: &DeviceBuffer<u32>,
        left_globals: &DeviceBuffer<f32>,
        right_globals: &DeviceBuffer<f32>,
        left_packed: &mut DeviceBuffer<u8>,
        right_packed: &mut DeviceBuffer<u8>,
        left_scales: &mut DeviceBuffer<u8>,
        right_scales: &mut DeviceBuffer<u8>,
        geometry: BucketQuantize,
    ) -> Result<()> {
        if geometry.ranked {
            return Err(Error::InvalidNvFp4("paired bucket quantization requires shared input"));
        }
        geometry.validate(
            input, selected, order, offsets, scale_offsets, left_globals, left_packed, left_scales,
        )?;
        geometry.validate(
            input, selected, order, offsets, scale_offsets, right_globals, right_packed,
            right_scales,
        )?;
        let launch = launch(geometry)?;
        Ok(self.quantize_pair.launch(
            stream,
            launch,
            (
                input,
                selected,
                order,
                offsets,
                scale_offsets,
                left_globals,
                right_globals,
                left_packed,
                right_packed,
                left_scales,
                right_scales,
                narrow(geometry.assignments)?,
                narrow(geometry.selected)?,
                narrow(geometry.input_rows)?,
                narrow(geometry.columns)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather_quantized(
        &self,
        stream: &Stream,
        selected: &DeviceBuffer<u32>,
        order: &DeviceBuffer<u32>,
        offsets: &DeviceBuffer<u32>,
        scale_offsets: &DeviceBuffer<u32>,
        source_packed: &DeviceBuffer<u8>,
        source_scales: &DeviceBuffer<u8>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        geometry: BucketQuantize,
    ) -> Result<()> {
        if geometry.ranked {
            return Err(Error::InvalidNvFp4("quantized bucket gather requires token input"));
        }
        require("bucket selections", geometry.assignments, selected.len())?;
        require("bucket order", geometry.assignments, order.len())?;
        require("bucket offsets", geometry.experts, offsets.len())?;
        require("bucket scale offsets", geometry.experts, scale_offsets.len())?;
        require(
            "unique packed input",
            product(geometry.input_rows, geometry.columns / 2)?,
            source_packed.len(),
        )?;
        require(
            "unique input scales",
            scale_elements(geometry.input_rows, geometry.columns)?,
            source_scales.len(),
        )?;
        require(
            "bucket packed input",
            product(geometry.assignments, geometry.columns / 2)?,
            packed.len(),
        )?;
        require(
            "bucket scales",
            scale_elements(padded_rows(geometry.assignments, geometry.experts)?, geometry.columns)?,
            scales.len(),
        )?;
        Ok(self.gather_quantized.launch(
            stream,
            LaunchConfig {
                grid: (narrow(geometry.assignments)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                selected,
                order,
                offsets,
                scale_offsets,
                source_packed,
                source_scales,
                packed,
                scales,
                narrow(geometry.assignments)?,
                narrow(geometry.selected)?,
                narrow(geometry.input_rows)?,
                narrow(geometry.columns)?,
            ),
        )?)
    }
}

impl BucketQuantize {
    #[allow(clippy::too_many_arguments)]
    fn validate(
        self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        order: &DeviceBuffer<u32>,
        offsets: &DeviceBuffer<u32>,
        scale_offsets: &DeviceBuffer<u32>,
        globals: &DeviceBuffer<f32>,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
    ) -> Result<()> {
        if self.assignments == 0 || self.experts == 0 || !self.columns.is_multiple_of(64) {
            return Err(Error::InvalidNvFp4("invalid bucket quantization geometry"));
        }
        require("bucket input", product(self.input_rows, self.columns)?, input.len())?;
        require("bucket selections", self.assignments, selected.len())?;
        require("bucket order", self.assignments, order.len())?;
        require("bucket offsets", self.experts, offsets.len())?;
        require("bucket scale offsets", self.experts, scale_offsets.len())?;
        require("bucket globals", self.experts, globals.len())?;
        require("bucket packed", product(self.assignments, self.columns / 2)?, packed.len())?;
        require(
            "bucket scales",
            scale_elements(padded_rows(self.assignments, self.experts)?, self.columns)?,
            scales.len(),
        )
    }
}

fn launch(geometry: BucketQuantize) -> Result<LaunchConfig> {
    const WARPS_PER_BLOCK: usize = 8;
    let warps = product(geometry.assignments, geometry.columns / 16)?;
    Ok(LaunchConfig {
        grid: (narrow(warps.div_ceil(WARPS_PER_BLOCK))?, 1, 1),
        block: (narrow(WARPS_PER_BLOCK * 32)?, 1, 1),
        shared_memory_bytes: 0,
    })
}

pub(super) fn padded_rows(assignments: usize, experts: usize) -> Result<usize> {
    let rows = assignments
        .checked_add(product(experts, 127)?)
        .ok_or(Error::InvalidNvFp4("bucketed scale capacity overflow"))?;
    Ok(rows / 128 * 128)
}
