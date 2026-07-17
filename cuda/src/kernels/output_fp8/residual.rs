use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{Fp8OutputKernels, Fp8ResidualWeightBuffers};
use crate::{Result, kernels::geometry::narrow};

impl Fp8OutputKernels {
    pub fn quantize_weight_residual(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<bf16>,
        buffers: &mut Fp8ResidualWeightBuffers<'_>,
    ) -> Result<()> {
        super::require("residual output source", self.spec.weight_elements()?, source.len())?;
        super::require(
            "residual output block scales",
            self.spec.weight_scale_elements()?,
            buffers.block_scales.len(),
        )?;
        self.validate_storage(
            buffers.weight,
            buffers.row_scales,
            buffers.residual,
            buffers.residual_scales,
        )?;
        self.quantize_weight(
            stream,
            source,
            buffers.weight,
            buffers.block_scales,
            buffers.row_scales,
        )?;
        Ok(self.quantize_residual.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(self.spec.output_features)?,
                    narrow(self.spec.input_features / 128)?,
                    1,
                ),
                block: (128, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                source,
                buffers.weight,
                buffers.row_scales,
                buffers.residual,
                buffers.residual_scales,
                narrow(self.spec.output_features)?,
                narrow(self.spec.input_features)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn project_residual(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        row_scales: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<u8>,
        residual_scales: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        super::require("residual output input", self.spec.input_features, input.len())?;
        self.validate_storage(weight, row_scales, residual, residual_scales)?;
        super::require("residual output logits", self.spec.output_features, output.len())?;
        Ok(self.residual.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.output_features.div_ceil(64))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weight,
                row_scales,
                residual,
                residual_scales,
                output,
                narrow(self.spec.output_features)?,
                narrow(self.spec.input_features)?,
            ),
        )?)
    }

    fn validate_storage(
        &self,
        weight: &DeviceBuffer<u8>,
        row_scales: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<u8>,
        residual_scales: &DeviceBuffer<f32>,
    ) -> Result<()> {
        super::require("residual output weight", self.spec.weight_elements()?, weight.len())?;
        super::require("residual output row scales", self.spec.output_features, row_scales.len())?;
        super::require(
            "residual output correction",
            self.spec.residual_elements()?,
            residual.len(),
        )?;
        super::require(
            "residual output scales",
            self.spec.residual_scale_elements()?,
            residual_scales.len(),
        )
    }
}
