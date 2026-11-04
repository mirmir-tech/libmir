//! A chunk that also leaves the state after each of its first tokens, for
//! speculative rollback: one scan writes the snapshots from its registers.

use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{MAMBA2_SCAN_ROWS, Mamba2Kernels, Mamba2Launch, launch::linear};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

impl Mamba2Kernels {
    /// Runs one chunk like [`Self::execute`], and writes the state after each
    /// of its first `histories.len()` tokens to consecutive states of
    /// `states` and the convolution history after it to `histories`.
    pub fn execute_snapshotted(
        &self,
        stream: &Stream,
        launch: &mut Mamba2Launch<'_>,
        states: &mut DeviceBuffer<f32>,
        histories: &mut [&mut DeviceBuffer<bf16>],
    ) -> Result<()> {
        self.validate(launch)?;
        let geometry = self.geometry;
        let count = histories.len();
        if count >= launch.tokens {
            return Err(Error::InvalidDecoderKernel("Mamba-2 snapshots cover the whole chunk"));
        }
        if states.len() < product(count, geometry.state_elements()?)? {
            return Err(Error::InvalidDecoderKernel("Mamba-2 snapshot states are too small"));
        }
        self.convolve(stream, launch)?;
        let channels = geometry.convolution_channels()?;
        for (index, history) in histories.iter_mut().enumerate() {
            require("Mamba-2 snapshot history", geometry.history_elements()?, history.len())?;
            self.history.launch(
                stream,
                linear(geometry.history_elements()?)?,
                (
                    launch.projection,
                    launch.history,
                    &mut **history,
                    narrow(index + 1)?,
                    narrow(channels)?,
                    narrow(geometry.convolution_kernel_size)?,
                    narrow(geometry.projection_width()?)?,
                    narrow(geometry.inner()?)?,
                ),
            )?;
        }
        let parameters = launch.parameters;
        self.scan_snapshots.launch(
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
                states,
                narrow(count)?,
                narrow(launch.tokens)?,
                narrow(geometry.heads)?,
                narrow(geometry.head_dim)?,
                narrow(geometry.state_size)?,
                narrow(geometry.groups)?,
                narrow(geometry.projection_width()?)?,
                narrow(geometry.inner()? + channels)?,
                geometry.time_step_minimum,
                geometry.time_step_maximum,
            ),
        )?;
        self.normalize(stream, launch)
    }
}
