use mircuda::{Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_kernel_files};

use super::{
    super::{
        affine::{AffineGemvSpec, compile_options},
        geometry::{narrow, product, require},
    },
    kernel::{SelectedReduceFallbackKernel, SelectedReduceInt4Kernel, SelectedReduceInt8Kernel},
};
use crate::{Error, Result};

/// Geometry for selected down projections and deterministic router reduction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SelectedAffineReduceSpec {
    pub matrix: AffineGemvSpec,
    pub expert_count: usize,
    pub selected_count: usize,
    pub tokens: usize,
}

impl SelectedAffineReduceSpec {
    pub const fn new(
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
    ) -> Result<Self> {
        Self::new_batch(matrix, expert_count, selected_count, 1)
    }

    pub const fn new_batch(
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
        tokens: usize,
    ) -> Result<Self> {
        if expert_count == 0 || selected_count == 0 || selected_count > expert_count || tokens == 0
        {
            return Err(Error::InvalidQuantizedGemv("invalid selected expert count"));
        }
        Ok(Self {
            matrix,
            expert_count,
            selected_count,
            tokens,
        })
    }
}

/// Device buffers for selected down projections and router reduction.
pub struct SelectedAffineReduceLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub selected: &'a DeviceBuffer<u32>,
    pub routing_weights: &'a DeviceBuffer<bf16>,
    pub weight: &'a DeviceBuffer<u32>,
    pub scales: &'a DeviceBuffer<bf16>,
    pub biases: &'a DeviceBuffer<bf16>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

#[derive(Clone, Debug)]
pub struct SelectedAffineReduce {
    kernel: ReduceKernel,
    spec: SelectedAffineReduceSpec,
}

#[derive(Clone, Debug)]
enum ReduceKernel {
    Int4(TypedKernel<SelectedReduceInt4Kernel>),
    Int8(TypedKernel<SelectedReduceInt8Kernel>),
    Fallback(TypedKernel<SelectedReduceFallbackKernel>),
}

impl SelectedAffineReduce {
    pub fn compile(compiler: &Compiler, spec: SelectedAffineReduceSpec) -> Result<Self> {
        let source = cuda_kernel_files!(
            "selected_affine_reduce_bf16.cu";
            "../../../kernels/affine_packed.cuh",
            "../../../kernels/selected_affine_reduce_bf16.cu",
        );
        let module = compiler.compile(source, &compile_options(spec.matrix.bits, true))?;
        let kernel = match spec.matrix.bits {
            4 => ReduceKernel::Int4(module.kernel()?),
            8 => ReduceKernel::Int8(module.kernel()?),
            2 | 3 | 5 | 6 => ReduceKernel::Fallback(module.kernel()?),
            _ => return Err(Error::InvalidQuantizedGemv("unsupported weight precision")),
        };
        Ok(Self { kernel, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        launch: &mut SelectedAffineReduceLaunch<'_>,
    ) -> Result<()> {
        self.validate(launch)?;
        let matrix = self.spec.matrix;
        let config = LaunchConfig {
            grid: (narrow(matrix.output_features.div_ceil(8))?, narrow(self.spec.tokens)?, 1),
            block: (32, 8, 1),
            shared_memory_bytes: 0,
        };
        let dimensions = (
            narrow(matrix.input_features)?,
            narrow(matrix.output_features)?,
            narrow(matrix.group_size)?,
            narrow(self.spec.expert_count)?,
            narrow(self.spec.selected_count)?,
        );
        Ok(match &self.kernel {
            ReduceKernel::Int4(kernel) => kernel.launch(
                stream,
                config,
                (
                    launch.input,
                    launch.selected,
                    launch.routing_weights,
                    launch.weight,
                    launch.scales,
                    launch.biases,
                    &mut *launch.output,
                    dimensions.0,
                    dimensions.1,
                    dimensions.2,
                    dimensions.3,
                    dimensions.4,
                ),
            ),
            ReduceKernel::Int8(kernel) => kernel.launch(
                stream,
                config,
                (
                    launch.input,
                    launch.selected,
                    launch.routing_weights,
                    launch.weight,
                    launch.scales,
                    launch.biases,
                    &mut *launch.output,
                    dimensions.0,
                    dimensions.1,
                    dimensions.2,
                    dimensions.3,
                    dimensions.4,
                ),
            ),
            ReduceKernel::Fallback(kernel) => kernel.launch(
                stream,
                config,
                (
                    launch.input,
                    launch.selected,
                    launch.routing_weights,
                    launch.weight,
                    launch.scales,
                    launch.biases,
                    &mut *launch.output,
                    dimensions.0,
                    dimensions.1,
                    dimensions.2,
                    dimensions.3,
                    dimensions.4,
                ),
            ),
        }?)
    }

    #[must_use]
    pub const fn spec(&self) -> SelectedAffineReduceSpec {
        self.spec
    }

    fn validate(&self, launch: &SelectedAffineReduceLaunch<'_>) -> Result<()> {
        let matrix = self.spec.matrix;
        let layout = matrix.layout()?;
        let selections = product(self.spec.selected_count, self.spec.tokens)?;
        let selected_input = product(matrix.input_features, selections)?;
        let packed = product(layout.packed_per_matrix, self.spec.expert_count)?;
        let grouped = product(layout.groups_per_matrix, self.spec.expert_count)?;
        require("selected input", selected_input, launch.input.len())?;
        require("selected experts", selections, launch.selected.len())?;
        require("routing weights", selections, launch.routing_weights.len())?;
        require("down weight", packed, launch.weight.len())?;
        require("down scales", grouped, launch.scales.len())?;
        require("down biases", grouped, launch.biases.len())?;
        require(
            "reduced output",
            product(matrix.output_features, self.spec.tokens)?,
            launch.output.len(),
        )
    }
}
