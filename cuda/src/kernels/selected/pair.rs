use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::{
    affine::AffineGemvSpec,
    geometry::{narrow, product, require},
};
use crate::{Error, Result};

cuda_export!(
    SelectedPairInt4Kernel = "libmir_cuda_selected_affine_pair_bf16_int4"(
        input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
        gate_weight: &DeviceBuffer<u32>, gate_scales: &DeviceBuffer<bf16>,
        gate_biases: &DeviceBuffer<bf16>, up_weight: &DeviceBuffer<u32>,
        up_scales: &DeviceBuffer<bf16>, up_biases: &DeviceBuffer<bf16>,
        gate_output: &mut DeviceBuffer<bf16>, up_output: &mut DeviceBuffer<bf16>,
        input_features: u32, output_features: u32, group_size: u32, expert_count: u32,
    )
);

cuda_export!(
    SelectedPairInt8Kernel = "libmir_cuda_selected_affine_pair_bf16_int8"(
        input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
        gate_weight: &DeviceBuffer<u32>, gate_scales: &DeviceBuffer<bf16>,
        gate_biases: &DeviceBuffer<bf16>, up_weight: &DeviceBuffer<u32>,
        up_scales: &DeviceBuffer<bf16>, up_biases: &DeviceBuffer<bf16>,
        gate_output: &mut DeviceBuffer<bf16>, up_output: &mut DeviceBuffer<bf16>,
        input_features: u32, output_features: u32, group_size: u32, expert_count: u32,
    )
);

/// Geometry for paired projections over a device-selected expert set.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SelectedAffinePairSpec {
    pub matrix: AffineGemvSpec,
    pub expert_count: usize,
    pub selected_count: usize,
}

impl SelectedAffinePairSpec {
    pub const fn new(
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
    ) -> Result<Self> {
        if expert_count == 0 || selected_count == 0 || selected_count > expert_count {
            return Err(Error::InvalidQuantizedGemv("invalid selected expert count"));
        }
        Ok(Self { matrix, expert_count, selected_count })
    }
}

/// Device buffers used by one selected-expert paired projection.
pub struct SelectedAffinePairLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub selected: &'a DeviceBuffer<u32>,
    pub gate_weight: &'a DeviceBuffer<u32>,
    pub gate_scales: &'a DeviceBuffer<bf16>,
    pub gate_biases: &'a DeviceBuffer<bf16>,
    pub up_weight: &'a DeviceBuffer<u32>,
    pub up_scales: &'a DeviceBuffer<bf16>,
    pub up_biases: &'a DeviceBuffer<bf16>,
    pub gate_output: &'a mut DeviceBuffer<bf16>,
    pub up_output: &'a mut DeviceBuffer<bf16>,
}

#[derive(Clone, Debug)]
pub struct SelectedAffinePair {
    kernel: PairKernel,
    spec: SelectedAffinePairSpec,
}

#[derive(Clone, Debug)]
enum PairKernel {
    Int4(TypedKernel<SelectedPairInt4Kernel>),
    Int8(TypedKernel<SelectedPairInt8Kernel>),
}

impl SelectedAffinePair {
    pub fn compile(compiler: &Compiler, spec: SelectedAffinePairSpec) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/selected_affine_pair_bf16.cu");
        let module =
            compiler.compile(source, &CompileOptions { fast_math: true, ..Default::default() })?;
        let kernel = match spec.matrix.bits {
            4 => PairKernel::Int4(module.kernel()?),
            8 => PairKernel::Int8(module.kernel()?),
            _ => return Err(Error::InvalidQuantizedGemv("unsupported weight precision")),
        };
        Ok(Self { kernel, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        launch: &mut SelectedAffinePairLaunch<'_>,
    ) -> Result<()> {
        self.validate(launch)?;
        let matrix = self.spec.matrix;
        let config = LaunchConfig {
            grid: (
                narrow(matrix.output_features.div_ceil(8))?,
                narrow(self.spec.selected_count)?,
                1,
            ),
            block: (32, 8, 1),
            shared_memory_bytes: 0,
        };
        let dimensions = (
            narrow(matrix.input_features)?,
            narrow(matrix.output_features)?,
            narrow(matrix.group_size)?,
            narrow(self.spec.expert_count)?,
        );
        Ok(match &self.kernel {
            PairKernel::Int4(kernel) => kernel.launch(
                stream,
                config,
                (
                    launch.input,
                    launch.selected,
                    launch.gate_weight,
                    launch.gate_scales,
                    launch.gate_biases,
                    launch.up_weight,
                    launch.up_scales,
                    launch.up_biases,
                    &mut *launch.gate_output,
                    &mut *launch.up_output,
                    dimensions.0,
                    dimensions.1,
                    dimensions.2,
                    dimensions.3,
                ),
            ),
            PairKernel::Int8(kernel) => kernel.launch(
                stream,
                config,
                (
                    launch.input,
                    launch.selected,
                    launch.gate_weight,
                    launch.gate_scales,
                    launch.gate_biases,
                    launch.up_weight,
                    launch.up_scales,
                    launch.up_biases,
                    &mut *launch.gate_output,
                    &mut *launch.up_output,
                    dimensions.0,
                    dimensions.1,
                    dimensions.2,
                    dimensions.3,
                ),
            ),
        }?)
    }

    #[must_use]
    pub const fn spec(&self) -> SelectedAffinePairSpec {
        self.spec
    }

    fn validate(&self, launch: &SelectedAffinePairLaunch<'_>) -> Result<()> {
        let matrix = self.spec.matrix;
        let layout = matrix.layout()?;
        let packed = product(layout.packed_per_matrix, self.spec.expert_count)?;
        let grouped = product(layout.groups_per_matrix, self.spec.expert_count)?;
        let output = product(matrix.output_features, self.spec.selected_count)?;
        require("input", matrix.input_features, launch.input.len())?;
        require("selected experts", self.spec.selected_count, launch.selected.len())?;
        require("gate weight", packed, launch.gate_weight.len())?;
        require("up weight", packed, launch.up_weight.len())?;
        for (name, actual) in [
            ("gate scales", launch.gate_scales.len()),
            ("gate biases", launch.gate_biases.len()),
            ("up scales", launch.up_scales.len()),
            ("up biases", launch.up_biases.len()),
        ] {
            require(name, grouped, actual)?;
        }
        require("gate output", output, launch.gate_output.len())?;
        require("up output", output, launch.up_output.len())
    }
}
