//! One decode token for each of several independent sequences whose states
//! are packed contiguously, one row per sequence.

use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16, cuda_export};

use super::{MAMBA2_SCAN_ROWS, Mamba2Kernels, Mamba2Parameters, launch::linear};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(
    pub(super) DecodeConvolutionKernel = "libmir_cuda_mamba2_decode_convolution_bf16"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>, bias: &DeviceBuffer<bf16>,
        histories: &mut DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, rows: u32,
        channels: u32, kernel_size: u32, input_stride: u32, input_offset: u32, has_bias: u32,
    )
);
cuda_export!(
    pub(super) DecodeScanKernel = "libmir_cuda_mamba2_decode_scan_bf16"(
        convolved: &DeviceBuffer<bf16>, projection: &DeviceBuffer<bf16>,
        a_log: &DeviceBuffer<bf16>, skip: &DeviceBuffer<bf16>, dt_bias: &DeviceBuffer<bf16>,
        states: &mut DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>, heads: u32,
        head_dim: u32, state_size: u32, groups: u32, projection_stride: u32, dt_offset: u32,
        dt_minimum: f32, dt_maximum: f32,
    )
);

/// One decode step of `rows` sequences. `states` and `histories` hold the
/// packed rows, advanced in place.
pub struct Mamba2DecodeBatch<'a> {
    pub rows: usize,
    pub projection: &'a DeviceBuffer<bf16>,
    pub parameters: Mamba2Parameters<'a>,
    pub states: &'a mut DeviceBuffer<f32>,
    pub histories: &'a mut DeviceBuffer<bf16>,
    /// Scratch for `rows * convolution_channels` convolved values.
    pub convolved: &'a mut DeviceBuffer<bf16>,
    /// Scratch for `rows * inner` scan outputs.
    pub scanned: &'a mut DeviceBuffer<bf16>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl Mamba2Kernels {
    /// Runs convolution, selective scan, and gated norm for one token of each
    /// packed row.
    pub fn execute_decode_batch(
        &self,
        stream: &Stream,
        launch: &mut Mamba2DecodeBatch<'_>,
    ) -> Result<()> {
        let geometry = self.geometry;
        let rows = launch.rows;
        if rows == 0 {
            return Err(Error::InvalidDecoderKernel("Mamba-2 decode batch has no rows"));
        }
        let (inner, channels) = (geometry.inner()?, geometry.convolution_channels()?);
        let width = geometry.projection_width()?;
        self.validate_parameters(launch.parameters)?;
        require("Mamba-2 batch projection", product(rows, width)?, launch.projection.len())?;
        require(
            "Mamba-2 batch states",
            product(rows, geometry.state_elements()?)?,
            launch.states.len(),
        )?;
        require(
            "Mamba-2 batch histories",
            product(rows, geometry.history_elements()?)?,
            launch.histories.len(),
        )?;
        require("Mamba-2 batch convolution", product(rows, channels)?, launch.convolved.len())?;
        require("Mamba-2 batch scan", product(rows, inner)?, launch.scanned.len())?;
        require("Mamba-2 batch output", product(rows, inner)?, launch.output.len())?;
        let parameters = launch.parameters;
        let bias = parameters.convolution_bias.unwrap_or(parameters.convolution_weight);
        self.decode_convolution.launch(
            stream,
            linear(product(rows, channels)?)?,
            (
                launch.projection,
                parameters.convolution_weight,
                bias,
                &mut *launch.histories,
                &mut *launch.convolved,
                narrow(rows)?,
                narrow(channels)?,
                narrow(geometry.convolution_kernel_size)?,
                narrow(width)?,
                narrow(inner)?,
                u32::from(parameters.convolution_bias.is_some()),
            ),
        )?;
        self.decode_scan.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(geometry.heads)?,
                    narrow(geometry.head_dim / MAMBA2_SCAN_ROWS)?,
                    narrow(rows)?,
                ),
                block: (narrow(geometry.state_size)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                &*launch.convolved,
                launch.projection,
                parameters.a_log,
                parameters.skip,
                parameters.dt_bias,
                &mut *launch.states,
                &mut *launch.scanned,
                narrow(geometry.heads)?,
                narrow(geometry.head_dim)?,
                narrow(geometry.state_size)?,
                narrow(geometry.groups)?,
                narrow(width)?,
                narrow(inner + channels)?,
                geometry.time_step_minimum,
                geometry.time_step_maximum,
            ),
        )?;
        self.gated_norm(
            stream,
            rows,
            launch.scanned,
            launch.projection,
            parameters.norm_weight,
            launch.output,
        )
    }
}
