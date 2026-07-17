use mircuda::{CompileOptions, Compiler, LaunchConfig, Stream, cuda_kernel_files};

use super::{
    NvFp4BankView, NvFp4MicroDownKernels, NvFp4MicroDownLaunch, NvFp4MicroSpec, activation,
    spec::require_len, validate_bank,
};
use crate::Result;

impl NvFp4MicroDownKernels {
    pub fn compile(compiler: &Compiler, spec: NvFp4MicroSpec) -> Result<Self> {
        spec.validate()?;
        let source = cuda_kernel_files!(
            "nvfp4_micro_down.cu";
            "../../../kernels/nvfp4_micro_common.cuh",
            "../../../kernels/nvfp4_micro_down.cu",
        );
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            gated_quantize: module.kernel()?,
            fc2: module.kernel()?,
            spec,
        })
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn execute(&self, stream: &Stream, launch: NvFp4MicroDownLaunch<'_>) -> Result<()> {
        self.validate(&launch)?;
        let spec = self.spec;
        let groups = spec.groups()?;
        self.gated_quantize.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(groups * spec.intermediate / 16)?, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                launch.gate,
                launch.up,
                launch.selected,
                launch.down.input_globals,
                &mut *launch.workspace.packed,
                &mut *launch.workspace.scales,
                u32::try_from(groups)?,
                u32::try_from(spec.intermediate)?,
                activation(spec.activation),
            ),
        )?;
        self.execute_prepared(
            stream,
            &*launch.workspace.packed,
            &*launch.workspace.scales,
            launch.selected,
            launch.routing,
            launch.down,
            launch.output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_prepared(
        &self,
        stream: &Stream,
        packed: &mircuda::DeviceBuffer<u8>,
        scales: &mircuda::DeviceBuffer<u8>,
        selected: &mircuda::DeviceBuffer<u32>,
        routing: &mircuda::DeviceBuffer<mircuda::bf16>,
        down: NvFp4BankView<'_>,
        output: &mut mircuda::DeviceBuffer<mircuda::bf16>,
    ) -> Result<()> {
        let spec = self.spec;
        Ok(self.fc2.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(spec.hidden.div_ceil(8))?, u32::try_from(spec.tokens)?, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                packed,
                scales,
                selected,
                routing,
                down.weight,
                down.scales,
                down.combined,
                output,
                u32::try_from(spec.intermediate)?,
                u32::try_from(spec.hidden)?,
                u32::try_from(spec.selected)?,
                u32::try_from(spec.tokens)?,
            ),
        )?)
    }

    fn validate(&self, launch: &NvFp4MicroDownLaunch<'_>) -> Result<()> {
        let spec = self.spec;
        let groups = spec.groups()?;
        require_len("micro down gate", groups * spec.intermediate, launch.gate.len())?;
        require_len("micro down up", groups * spec.intermediate, launch.up.len())?;
        require_len("micro down selected", groups, launch.selected.len())?;
        require_len("micro down routing", groups, launch.routing.len())?;
        require_len(
            "micro down packed",
            groups * spec.intermediate / 2,
            launch.workspace.packed.len(),
        )?;
        require_len(
            "micro down scales",
            groups * spec.intermediate / 16,
            launch.workspace.scales.len(),
        )?;
        require_len("micro down output", spec.tokens * spec.hidden, launch.output.len())?;
        validate_bank(launch.down, spec.experts, spec.intermediate, spec.hidden)
    }
}
