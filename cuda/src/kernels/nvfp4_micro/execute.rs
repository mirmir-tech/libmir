use mircuda::Stream;

use super::{
    NvFp4MicroGateLaunch, NvFp4MicroGateWorkspace, NvFp4MicroKernels, NvFp4MicroLaunch,
    spec::require_len, validate_bank,
};
use crate::Result;

impl NvFp4MicroKernels {
    #[allow(clippy::needless_pass_by_value)]
    pub fn execute(&self, stream: &Stream, launch: NvFp4MicroLaunch<'_>) -> Result<()> {
        self.validate(&launch)?;
        let workspace = launch.workspace;
        self.execute_gate(
            stream,
            NvFp4MicroGateLaunch {
                input: launch.input,
                selected: launch.selected,
                banks: launch.banks,
                workspace: NvFp4MicroGateWorkspace {
                    gate_packed: &mut *workspace.gate_packed,
                    up_packed: &mut *workspace.up_packed,
                    gate_scales: &mut *workspace.gate_scales,
                    up_scales: &mut *workspace.up_scales,
                    output_packed: &mut *workspace.intermediate_packed,
                    output_scales: &mut *workspace.intermediate_scales,
                },
                output_scale_stride: 0,
            },
        )?;
        self.down.execute_prepared(
            stream,
            &*workspace.intermediate_packed,
            &*workspace.intermediate_scales,
            launch.selected,
            launch.routing,
            launch.banks.down,
            launch.output,
        )
    }

    fn validate(&self, launch: &NvFp4MicroLaunch<'_>) -> Result<()> {
        let spec = self.spec;
        let groups = spec.groups()?;
        require_len("micro input", spec.tokens * spec.hidden, launch.input.len())?;
        require_len("micro selected", groups, launch.selected.len())?;
        require_len("micro routing", groups, launch.routing.len())?;
        require_len("micro output", spec.tokens * spec.hidden, launch.output.len())?;
        require_len(
            "micro intermediate packed",
            groups * spec.intermediate / 2,
            launch.workspace.intermediate_packed.len(),
        )?;
        require_len(
            "micro intermediate scales",
            groups * spec.intermediate / 16,
            launch.workspace.intermediate_scales.len(),
        )?;
        validate_bank(launch.banks.down, spec.experts, spec.intermediate, spec.hidden)
    }
}
