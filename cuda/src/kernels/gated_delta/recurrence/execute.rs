use mircuda::{LaunchConfig, Stream};

use super::{GatedDeltaLaunch, GatedDeltaRecurrence};
use crate::{
    Result,
    kernels::geometry::{narrow, product, require},
};

const WARPS: usize = 4;

macro_rules! launch_value_tiled {
    ($method:ident, $kernel:ident, $tile:literal) => {
        pub(super) fn $method(
            &self,
            stream: &Stream,
            launch: &mut GatedDeltaLaunch<'_>,
        ) -> Result<()> {
            Ok(self.$kernel.launch(
                stream,
                self.config(WARPS, $tile)?,
                (
                    launch.query,
                    launch.key,
                    launch.value,
                    &*launch.decay,
                    &*launch.update,
                    &mut *launch.state,
                    &mut *launch.output,
                    narrow(self.spec.tokens)?,
                    narrow(self.spec.key_heads)?,
                    narrow(self.spec.value_heads)?,
                    narrow(self.spec.key_dim)?,
                    narrow(self.spec.value_dim)?,
                ),
            )?)
        }
    };
}

impl GatedDeltaRecurrence {
    launch_value_tiled!(launch_value_tiled_2, value_tiled_2, 2);

    launch_value_tiled!(launch_value_tiled_4, value_tiled_4, 4);

    launch_value_tiled!(launch_value_tiled_8, value_tiled_8, 8);

    pub(super) fn validate_launch(&self, launch: &GatedDeltaLaunch<'_>) -> Result<()> {
        let key = product(product(self.spec.tokens, self.spec.key_heads)?, self.spec.key_dim)?;
        let value =
            product(product(self.spec.tokens, self.spec.value_heads)?, self.spec.value_dim)?;
        let gates = product(self.spec.tokens, self.spec.value_heads)?;
        require("Gated Delta query", key, launch.query.len())?;
        require("Gated Delta key", key, launch.key.len())?;
        require("Gated Delta value", value, launch.value.len())?;
        require("Gated Delta alpha", gates, launch.alpha.len())?;
        require("Gated Delta beta", gates, launch.beta.len())?;
        require("Gated Delta A log", self.spec.value_heads, launch.a_log.len())?;
        require("Gated Delta time bias", self.spec.value_heads, launch.dt_bias.len())?;
        require("Gated Delta decay", gates, launch.decay.len())?;
        require("Gated Delta update", gates, launch.update.len())?;
        require("Gated Delta state", self.state_elements()?, launch.state.len())?;
        require("Gated Delta output", value, launch.output.len())
    }

    pub(super) fn prepare_parameters(
        &self,
        stream: &Stream,
        launch: &mut GatedDeltaLaunch<'_>,
    ) -> Result<()> {
        if self.spec.tokens == 1 {
            return Ok(());
        }
        let gates = product(self.spec.tokens, self.spec.value_heads)?;
        Ok(self.parameters.launch(
            stream,
            LaunchConfig {
                grid: (narrow(gates.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                launch.alpha,
                launch.beta,
                launch.a_log,
                launch.dt_bias,
                &mut *launch.decay,
                &mut *launch.update,
                narrow(self.spec.tokens)?,
                narrow(self.spec.value_heads)?,
            ),
        )?)
    }

    pub(super) fn launch_serial(
        &self,
        stream: &Stream,
        launch: &mut GatedDeltaLaunch<'_>,
    ) -> Result<()> {
        Ok(self.serial.launch(
            stream,
            self.config(WARPS, 1)?,
            (
                launch.query,
                launch.key,
                launch.value,
                launch.alpha,
                launch.beta,
                launch.a_log,
                launch.dt_bias,
                &*launch.decay,
                &*launch.update,
                &mut *launch.state,
                &mut *launch.output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.key_heads)?,
                narrow(self.spec.value_heads)?,
                narrow(self.spec.key_dim)?,
                narrow(self.spec.value_dim)?,
            ),
        )?)
    }

    fn config(&self, warps: usize, value_tile: usize) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            grid: (
                1,
                narrow(self.spec.value_dim.div_ceil(warps * value_tile))?,
                narrow(self.spec.value_heads)?,
            ),
            block: (32, narrow(warps)?, 1),
            shared_memory_bytes: 0,
        })
    }
}
