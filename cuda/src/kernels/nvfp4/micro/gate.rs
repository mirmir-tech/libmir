use mircuda::{LaunchConfig, Stream};

use super::{
    NvFp4MicroGateLaunch, NvFp4MicroKernels, activation, spec::require_len, validate_bank,
};
use crate::Result;

impl NvFp4MicroKernels {
    #[allow(clippy::needless_pass_by_value)]
    pub fn execute_gate(&self, stream: &Stream, launch: NvFp4MicroGateLaunch<'_>) -> Result<()> {
        self.validate_gate(&launch)?;
        let spec = self.spec;
        let groups = spec.groups()?;
        self.quantize_pair.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(groups * spec.hidden / 16)?, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                launch.input,
                launch.selected,
                launch.banks.gate.input_globals,
                launch.banks.up.input_globals,
                &mut *launch.workspace.gate_packed,
                &mut *launch.workspace.up_packed,
                &mut *launch.workspace.gate_scales,
                &mut *launch.workspace.up_scales,
                u32::try_from(groups)?,
                u32::try_from(spec.selected)?,
                u32::try_from(spec.hidden)?,
            ),
        )?;
        Ok(self.fc1.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(groups * spec.intermediate / 16)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                &*launch.workspace.gate_packed,
                &*launch.workspace.gate_scales,
                &*launch.workspace.up_packed,
                &*launch.workspace.up_scales,
                launch.selected,
                launch.banks.gate.weight,
                launch.banks.gate.scales,
                launch.banks.gate.combined,
                launch.banks.up.weight,
                launch.banks.up.scales,
                launch.banks.up.combined,
                launch.banks.down.input_globals,
                &mut *launch.workspace.output_packed,
                &mut *launch.workspace.output_scales,
                u32::try_from(groups)?,
                u32::try_from(spec.hidden)?,
                u32::try_from(spec.intermediate)?,
                u32::try_from(launch.output_scale_stride)?,
                activation(spec.activation),
            ),
        )?)
    }

    fn validate_gate(&self, launch: &NvFp4MicroGateLaunch<'_>) -> Result<()> {
        let spec = self.spec;
        let groups = spec.groups()?;
        require_len("micro gate input", spec.tokens * spec.hidden, launch.input.len())?;
        require_len("micro gate selected", groups, launch.selected.len())?;
        require_len(
            "micro gate packed",
            groups * spec.hidden / 2,
            launch.workspace.gate_packed.len(),
        )?;
        require_len("micro up packed", groups * spec.hidden / 2, launch.workspace.up_packed.len())?;
        require_len(
            "micro gate scales",
            groups * spec.hidden / 16,
            launch.workspace.gate_scales.len(),
        )?;
        require_len(
            "micro up scales",
            groups * spec.hidden / 16,
            launch.workspace.up_scales.len(),
        )?;
        require_len(
            "micro gate output",
            groups * spec.intermediate / 2,
            launch.workspace.output_packed.len(),
        )?;
        let scales = if launch.output_scale_stride == 0 {
            groups * spec.intermediate / 16
        } else {
            groups * launch.output_scale_stride
        };
        require_len("micro gate output scales", scales, launch.workspace.output_scales.len())?;
        validate_bank(launch.banks.gate, spec.experts, spec.hidden, spec.intermediate)?;
        validate_bank(launch.banks.up, spec.experts, spec.hidden, spec.intermediate)
    }
}
