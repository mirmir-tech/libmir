use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{BucketQuantize, NvFp4BucketPreparation, quantize::padded_rows};
use crate::{
    Error, GatedActivation, Result,
    kernels::{
        geometry::{narrow, product, require},
        scale_elements,
    },
};

impl NvFp4BucketPreparation {
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_gated(
        &self,
        stream: &Stream,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        order: &DeviceBuffer<u32>,
        offsets: &DeviceBuffer<u32>,
        scale_offsets: &DeviceBuffer<u32>,
        global_scales: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        geometry: BucketQuantize,
        activation: GatedActivation,
    ) -> Result<()> {
        const WARPS_PER_BLOCK: usize = 8;
        if !geometry.ranked || !geometry.columns.is_multiple_of(64) {
            return Err(Error::InvalidNvFp4("invalid gated bucket quantization geometry"));
        }
        let elements = product(geometry.assignments, geometry.columns)?;
        require("gated bucket gate", elements, gate.len())?;
        require("gated bucket up", elements, up.len())?;
        require("gated bucket selections", geometry.assignments, selected.len())?;
        require("gated bucket order", geometry.assignments, order.len())?;
        require("gated bucket offsets", geometry.experts, offsets.len())?;
        require("gated bucket scale offsets", geometry.experts, scale_offsets.len())?;
        require("gated bucket globals", geometry.experts, global_scales.len())?;
        require("gated bucket packed", elements / 2, packed.len())?;
        require(
            "gated bucket scales",
            scale_elements(padded_rows(geometry.assignments, geometry.experts)?, geometry.columns)?,
            scales.len(),
        )?;
        let warps = product(geometry.assignments, geometry.columns / 32)?;
        Ok(self.gated_quantize.launch(
            stream,
            LaunchConfig {
                grid: (narrow(warps.div_ceil(WARPS_PER_BLOCK))?, 1, 1),
                block: (narrow(WARPS_PER_BLOCK * 32)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                gate,
                up,
                selected,
                order,
                offsets,
                scale_offsets,
                global_scales,
                packed,
                scales,
                narrow(geometry.assignments)?,
                narrow(geometry.columns)?,
                activation.code(),
            ),
        )?)
    }
}
