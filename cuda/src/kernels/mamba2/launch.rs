use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{MAMBA2_SCAN_ROWS, Mamba2Kernels};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

/// Per-layer parameters of one Mamba-2 mixer.
#[derive(Clone, Copy)]
pub struct Mamba2Parameters<'a> {
    pub convolution_weight: &'a DeviceBuffer<bf16>,
    /// Convolution bias, or `None` when the checkpoint has none.
    pub convolution_bias: Option<&'a DeviceBuffer<bf16>>,
    pub a_log: &'a DeviceBuffer<bf16>,
    pub skip: &'a DeviceBuffer<bf16>,
    pub dt_bias: &'a DeviceBuffer<bf16>,
    pub norm_weight: &'a DeviceBuffer<bf16>,
}

/// One chunk of one sequence through the mixer. `projection` holds `tokens`
/// rows of the fused input projection; `output` receives `tokens * inner`
/// gated, normalized values for the output projection.
pub struct Mamba2Launch<'a> {
    pub tokens: usize,
    pub projection: &'a DeviceBuffer<bf16>,
    pub parameters: Mamba2Parameters<'a>,
    pub state: &'a mut DeviceBuffer<f32>,
    pub history: &'a DeviceBuffer<bf16>,
    pub next_history: &'a mut DeviceBuffer<bf16>,
    /// Scratch for `tokens * convolution_channels` convolved values.
    pub convolved: &'a mut DeviceBuffer<bf16>,
    /// Scratch for `tokens * inner` scan outputs.
    pub scanned: &'a mut DeviceBuffer<bf16>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl Mamba2Kernels {
    /// Runs convolution, selective scan, and gated norm for one chunk.
    pub fn execute(&self, stream: &Stream, launch: &mut Mamba2Launch<'_>) -> Result<()> {
        self.validate(launch)?;
        self.convolve(stream, launch)?;
        self.scan(stream, launch)?;
        self.normalize(stream, launch)
    }

    pub(super) fn validate(&self, launch: &Mamba2Launch<'_>) -> Result<()> {
        let geometry = self.geometry;
        if launch.tokens == 0 {
            return Err(Error::InvalidDecoderKernel("Mamba-2 chunk has no tokens"));
        }
        let tokens = launch.tokens;
        let inner = geometry.inner()?;
        let channels = geometry.convolution_channels()?;
        require(
            "Mamba-2 projection",
            product(tokens, geometry.projection_width()?)?,
            launch.projection.len(),
        )?;
        self.validate_parameters(launch.parameters)?;
        require("Mamba-2 state", geometry.state_elements()?, launch.state.len())?;
        require("Mamba-2 history", geometry.history_elements()?, launch.history.len())?;
        require("Mamba-2 next history", geometry.history_elements()?, launch.next_history.len())?;
        require("Mamba-2 convolution output", product(tokens, channels)?, launch.convolved.len())?;
        require("Mamba-2 scan output", product(tokens, inner)?, launch.scanned.len())?;
        require("Mamba-2 output", product(tokens, inner)?, launch.output.len())
    }

    pub(super) fn validate_parameters(&self, parameters: Mamba2Parameters<'_>) -> Result<()> {
        let geometry = self.geometry;
        let channels = geometry.convolution_channels()?;
        require(
            "Mamba-2 convolution weight",
            product(channels, geometry.convolution_kernel_size)?,
            parameters.convolution_weight.len(),
        )?;
        if let Some(bias) = parameters.convolution_bias {
            require("Mamba-2 convolution bias", channels, bias.len())?;
        }
        for (operand, buffer) in [
            ("Mamba-2 A_log", parameters.a_log),
            ("Mamba-2 D", parameters.skip),
            ("Mamba-2 dt_bias", parameters.dt_bias),
        ] {
            require(operand, geometry.heads, buffer.len())?;
        }
        require("Mamba-2 norm weight", geometry.inner()?, parameters.norm_weight.len())
    }

    pub(super) fn convolve(&self, stream: &Stream, launch: &mut Mamba2Launch<'_>) -> Result<()> {
        let geometry = self.geometry;
        let channels = geometry.convolution_channels()?;
        let arguments = (
            narrow(launch.tokens)?,
            narrow(channels)?,
            narrow(geometry.convolution_kernel_size)?,
            narrow(geometry.projection_width()?)?,
            narrow(geometry.inner()?)?,
        );
        let parameters = launch.parameters;
        let bias = parameters.convolution_bias.unwrap_or(parameters.convolution_weight);
        self.convolution.launch(
            stream,
            linear(product(launch.tokens, channels)?)?,
            (
                launch.projection,
                parameters.convolution_weight,
                bias,
                launch.history,
                &mut *launch.convolved,
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
                u32::from(parameters.convolution_bias.is_some()),
            ),
        )?;
        Ok(self.history.launch(
            stream,
            linear(geometry.history_elements()?)?,
            (
                launch.projection,
                launch.history,
                &mut *launch.next_history,
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
            ),
        )?)
    }

    fn scan(&self, stream: &Stream, launch: &mut Mamba2Launch<'_>) -> Result<()> {
        let geometry = self.geometry;
        let parameters = launch.parameters;
        Ok(self.scan.launch(
            stream,
            LaunchConfig {
                grid: (narrow(geometry.heads)?, narrow(geometry.head_dim / MAMBA2_SCAN_ROWS)?, 1),
                block: (narrow(geometry.state_size)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                &*launch.convolved,
                launch.projection,
                parameters.a_log,
                parameters.skip,
                parameters.dt_bias,
                &mut *launch.state,
                &mut *launch.scanned,
                narrow(launch.tokens)?,
                narrow(geometry.heads)?,
                narrow(geometry.head_dim)?,
                narrow(geometry.state_size)?,
                narrow(geometry.groups)?,
                narrow(geometry.projection_width()?)?,
                narrow(geometry.inner()? + geometry.convolution_channels()?)?,
                geometry.time_step_minimum,
                geometry.time_step_maximum,
            ),
        )?)
    }

    pub(super) fn normalize(&self, stream: &Stream, launch: &mut Mamba2Launch<'_>) -> Result<()> {
        self.gated_norm(
            stream,
            launch.tokens,
            launch.scanned,
            launch.projection,
            launch.parameters.norm_weight,
            launch.output,
        )
    }

    pub(super) fn gated_norm(
        &self,
        stream: &Stream,
        tokens: usize,
        scanned: &DeviceBuffer<bf16>,
        projection: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let geometry = self.geometry;
        let inner = geometry.inner()?;
        Ok(self.gated_norm.launch(
            stream,
            LaunchConfig {
                grid: (narrow(product(tokens, geometry.groups)?)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                scanned,
                projection,
                weight,
                output,
                narrow(tokens)?,
                narrow(inner)?,
                narrow(inner / geometry.groups)?,
                narrow(geometry.projection_width()?)?,
                geometry.norm_epsilon,
            ),
        )?)
    }
}

pub(super) fn linear(values: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(values.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
