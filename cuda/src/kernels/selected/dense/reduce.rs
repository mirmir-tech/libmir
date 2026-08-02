use mircuda::Stream;

use super::{SelectedDenseMoe, SelectedDenseReduceLaunch, launch};
use crate::{Result, kernels::geometry::narrow};

impl SelectedDenseMoe {
    pub fn reduce(
        &self,
        stream: &Stream,
        launch: &mut SelectedDenseReduceLaunch<'_>,
    ) -> Result<()> {
        self.validate_reduce(launch)?;
        if self.spec.down_transposed && self.spec.tokens == 1 {
            return self.reduce_split(stream, launch);
        }
        let spec = self.spec;
        Ok(self.reduce.launch(
            stream,
            launch::reduce(spec)?,
            (
                launch.input,
                launch.selected,
                launch.routing,
                launch.weight,
                launch.bias,
                &mut *launch.output,
                narrow(spec.output_features)?,
                narrow(spec.input_features)?,
                narrow(spec.expert_count)?,
                narrow(spec.selected_count)?,
                u32::from(spec.down_transposed),
                u32::from(spec.down_bias),
            ),
        )?)
    }

    fn reduce_split(
        &self,
        stream: &Stream,
        launch: &mut SelectedDenseReduceLaunch<'_>,
    ) -> Result<()> {
        let spec = self.spec;
        self.project.launch(
            stream,
            launch::project(spec)?,
            (
                launch.input,
                launch.selected,
                launch.routing,
                launch.weight,
                launch.bias,
                &mut *launch.partial,
                narrow(spec.output_features)?,
                narrow(spec.input_features)?,
                narrow(spec.expert_count)?,
                narrow(spec.selected_count)?,
                u32::from(spec.down_bias),
                u32::from(spec.down_transposed),
            ),
        )?;
        self.finalize(stream, launch)
    }

    pub(super) fn finalize(
        &self,
        stream: &Stream,
        launch: &mut SelectedDenseReduceLaunch<'_>,
    ) -> Result<()> {
        let spec = self.spec;
        Ok(self.finalize.launch(
            stream,
            launch::finalize(spec)?,
            (
                &*launch.partial,
                &mut *launch.output,
                narrow(spec.input_features)?,
                narrow(spec.selected_count)?,
            ),
        )?)
    }
}
