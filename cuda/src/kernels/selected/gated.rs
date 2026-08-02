use mircuda::{
    Compiler, DeviceBuffer, KernelSignature, LaunchConfig, Stream, TypedKernel, bf16,
    cuda_kernel_files,
};

use super::{
    super::{
        affine::{AffineGemvSpec, compile_options},
        geometry::{narrow, product, require},
    },
    kernel::{SelectedGatedFallbackKernel, SelectedGatedInt4Kernel, SelectedGatedInt8Kernel},
};
use crate::{Error, Result};

/// Gated MLP activation applied after separately rounded BF16 projections.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GatedActivation {
    GeluTanh,
    Silu,
}

impl GatedActivation {
    pub(crate) const fn code(self) -> u32 {
        match self {
            Self::GeluTanh => 0,
            Self::Silu => 1,
        }
    }
}

/// Geometry for selected paired projections followed by a gated activation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SelectedAffineGatedSpec {
    pub matrix: AffineGemvSpec,
    pub expert_count: usize,
    pub selected_count: usize,
    pub tokens: usize,
    pub activation: GatedActivation,
}

impl SelectedAffineGatedSpec {
    pub const fn new(
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
        activation: GatedActivation,
    ) -> Result<Self> {
        Self::new_batch(matrix, expert_count, selected_count, 1, activation)
    }

    pub const fn new_batch(
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
        tokens: usize,
        activation: GatedActivation,
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
            activation,
        })
    }
}

/// Device buffers for one selected gated projection.
pub struct SelectedAffineGatedLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub selected: &'a DeviceBuffer<u32>,
    pub gate_weight: &'a DeviceBuffer<u32>,
    pub gate_scales: &'a DeviceBuffer<bf16>,
    pub gate_biases: &'a DeviceBuffer<bf16>,
    pub up_weight: &'a DeviceBuffer<u32>,
    pub up_scales: &'a DeviceBuffer<bf16>,
    pub up_biases: &'a DeviceBuffer<bf16>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

#[derive(Clone, Debug)]
pub struct SelectedAffineGated {
    kernel: GatedKernel,
    spec: SelectedAffineGatedSpec,
}

#[derive(Clone, Debug)]
enum GatedKernel {
    Int4(TypedKernel<SelectedGatedInt4Kernel>),
    Int8(TypedKernel<SelectedGatedInt8Kernel>),
    Fallback(TypedKernel<SelectedGatedFallbackKernel>),
}

impl SelectedAffineGated {
    pub fn compile(compiler: &Compiler, spec: SelectedAffineGatedSpec) -> Result<Self> {
        let source = cuda_kernel_files!(
            "selected_affine_gated_bf16.cu";
            "../../../kernels/affine_packed.cuh",
            "../../../kernels/selected_affine_gated_bf16.cu",
        );
        let module = compiler.compile(source, &compile_options(spec.matrix.bits, true))?;
        let kernel = match spec.matrix.bits {
            4 => GatedKernel::Int4(module.kernel()?),
            8 => GatedKernel::Int8(module.kernel()?),
            2 | 3 | 5 | 6 => GatedKernel::Fallback(module.kernel()?),
            _ => return Err(Error::InvalidQuantizedGemv("unsupported weight precision")),
        };
        Ok(Self { kernel, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        launch: &mut SelectedAffineGatedLaunch<'_>,
    ) -> Result<()> {
        self.validate(launch)?;
        let matrix = self.spec.matrix;
        let config = LaunchConfig {
            grid: (
                narrow(matrix.output_features.div_ceil(8))?,
                narrow(self.spec.selected_count)?,
                narrow(self.spec.tokens)?,
            ),
            block: (32, 8, 1),
            shared_memory_bytes: 0,
        };
        let dimensions = (
            narrow(matrix.input_features)?,
            narrow(matrix.output_features)?,
            narrow(matrix.group_size)?,
            narrow(self.spec.expert_count)?,
            self.spec.activation.code(),
        );
        match &self.kernel {
            GatedKernel::Int4(kernel) => launch_kernel(kernel, stream, config, launch, dimensions),
            GatedKernel::Int8(kernel) => launch_kernel(kernel, stream, config, launch, dimensions),
            GatedKernel::Fallback(kernel) => {
                launch_kernel(kernel, stream, config, launch, dimensions)
            },
        }
    }

    #[must_use]
    pub const fn spec(&self) -> SelectedAffineGatedSpec {
        self.spec
    }

    fn validate(&self, launch: &SelectedAffineGatedLaunch<'_>) -> Result<()> {
        let matrix = self.spec.matrix;
        let layout = matrix.layout()?;
        let packed = product(layout.packed_per_matrix, self.spec.expert_count)?;
        let grouped = product(layout.groups_per_matrix, self.spec.expert_count)?;
        require("input", product(matrix.input_features, self.spec.tokens)?, launch.input.len())?;
        require(
            "selected experts",
            product(self.spec.selected_count, self.spec.tokens)?,
            launch.selected.len(),
        )?;
        for (name, actual) in
            [("gate weight", launch.gate_weight.len()), ("up weight", launch.up_weight.len())]
        {
            require(name, packed, actual)?;
        }
        for (name, actual) in [
            ("gate scales", launch.gate_scales.len()),
            ("gate biases", launch.gate_biases.len()),
            ("up scales", launch.up_scales.len()),
            ("up biases", launch.up_biases.len()),
        ] {
            require(name, grouped, actual)?;
        }
        require(
            "gated output",
            product(product(matrix.output_features, self.spec.selected_count)?, self.spec.tokens)?,
            launch.output.len(),
        )
    }
}

type GatedDimensions = (u32, u32, u32, u32, u32);

fn launch_kernel<S>(
    kernel: &TypedKernel<S>,
    stream: &Stream,
    config: LaunchConfig,
    launch: &mut SelectedAffineGatedLaunch<'_>,
    dimensions: GatedDimensions,
) -> Result<()>
where
    for<'a> S: KernelSignature<
        Arguments<'a> = (
            &'a DeviceBuffer<bf16>,
            &'a DeviceBuffer<u32>,
            &'a DeviceBuffer<u32>,
            &'a DeviceBuffer<bf16>,
            &'a DeviceBuffer<bf16>,
            &'a DeviceBuffer<u32>,
            &'a DeviceBuffer<bf16>,
            &'a DeviceBuffer<bf16>,
            &'a mut DeviceBuffer<bf16>,
            u32,
            u32,
            u32,
            u32,
            u32,
        ),
    >,
{
    Ok(kernel.launch(
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
            &mut *launch.output,
            dimensions.0,
            dimensions.1,
            dimensions.2,
            dimensions.3,
            dimensions.4,
        ),
    )?)
}
