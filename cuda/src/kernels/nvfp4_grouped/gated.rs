use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{GroupedQuantize, NvFp4GroupedPreparation, narrow, product, scale_elements};
use crate::{Error, GatedActivation, Result};

impl NvFp4GroupedPreparation {
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_gated(
        &self,
        stream: &Stream,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        global_scales: &DeviceBuffer<f32>,
        packed: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        geometry: GroupedQuantize,
        activation: GatedActivation,
    ) -> Result<()> {
        if !geometry.ranked {
            return Err(Error::InvalidNvFp4("gated quantization requires ranked inputs"));
        }
        geometry.validate(gate, selected, global_scales, packed, scales)?;
        geometry.validate(up, selected, global_scales, packed, scales)?;
        let blocks = product(geometry.groups, geometry.columns / 16)?;
        let scale_stride = scale_elements(1, geometry.columns)?;
        Ok(self.gated_quantize.launch(
            stream,
            LaunchConfig {
                grid: (narrow(blocks)?, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                gate,
                up,
                selected,
                global_scales,
                packed,
                scales,
                narrow(geometry.groups)?,
                narrow(geometry.columns)?,
                narrow(scale_stride)?,
                activation.code(),
            ),
        )?)
    }
}
