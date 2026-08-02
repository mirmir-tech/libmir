use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::narrow;
use crate::{Error, Result};

mod canonicalize;
mod expert_major;
mod launch;
mod reduce;
mod tensor_core;
mod validation;

pub use canonicalize::DenseExpertCanonicalizer;
use expert_major::ExpertMajorKernels;
pub use expert_major::SelectedDenseDispatch;
use tensor_core::TensorCoreKernels;

cuda_export!(
    DenseGatedKernel = "libmir_cuda_selected_dense_gated_bf16"(
        input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
        gate_weight: &DeviceBuffer<bf16>, gate_bias: &DeviceBuffer<bf16>,
        up_weight: &DeviceBuffer<bf16>, up_bias: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, input_features: u32, output_features: u32,
        expert_count: u32, selected_count: u32, gate_up_layout: u32,
        gate_transposed: u32, up_transposed: u32, has_gate_bias: u32,
        has_up_bias: u32, activation: u32, alpha: f32, limit: f32, up_shift: f32,
    )
);

cuda_export!(
    DenseReduceKernel = "libmir_cuda_selected_dense_reduce_bf16"(
        input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        input_features: u32, output_features: u32, expert_count: u32,
        selected_count: u32, transposed: u32, has_bias: u32,
    )
);
cuda_export!(
    DenseProjectKernel = "libmir_cuda_selected_dense_project_bf16"(
        input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>, partial: &mut DeviceBuffer<f32>,
        input_features: u32, output_features: u32, expert_count: u32,
        selected_count: u32, has_bias: u32, transposed: u32,
    )
);
cuda_export!(
    DenseFinalizeKernel = "libmir_cuda_selected_dense_finalize_bf16"(
        partial: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>,
        output_features: u32, selected_count: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseGateUpLayout {
    Separate,
    FusedContiguous,
    FusedInterleaved,
}

impl DenseGateUpLayout {
    const fn code(self) -> u32 {
        match self {
            Self::Separate => 0,
            Self::FusedContiguous => 1,
            Self::FusedInterleaved => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseGatedActivation {
    pub kind: u32,
    pub alpha: f32,
    pub limit: f32,
    pub up_shift: f32,
}

impl DenseGatedActivation {
    pub const GELU_TANH: Self = Self {
        kind: 0,
        alpha: 1.0,
        limit: 0.0,
        up_shift: 0.0,
    };
    pub const SILU: Self = Self {
        kind: 1,
        alpha: 1.0,
        limit: 0.0,
        up_shift: 0.0,
    };

    #[must_use]
    pub const fn clamped_silu(alpha: f32, limit: f32, up_shift: f32) -> Self {
        Self { kind: 2, alpha, limit, up_shift }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct SelectedDenseMoeSpec {
    pub tokens: usize,
    pub input_features: usize,
    pub output_features: usize,
    pub expert_count: usize,
    pub selected_count: usize,
    pub gate_up_layout: DenseGateUpLayout,
    pub gate_transposed: bool,
    pub up_transposed: bool,
    pub down_transposed: bool,
    pub gate_bias: bool,
    pub up_bias: bool,
    pub down_bias: bool,
    pub activation: DenseGatedActivation,
}

impl SelectedDenseMoeSpec {
    pub const fn validate(self) -> Result<Self> {
        if self.tokens == 0
            || self.input_features == 0
            || self.output_features == 0
            || self.expert_count == 0
            || self.selected_count == 0
            || self.selected_count > self.expert_count
        {
            return Err(Error::InvalidDecoderKernel("invalid dense selected-expert geometry"));
        }
        Ok(self)
    }
}

pub struct SelectedDenseGateLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub selected: &'a DeviceBuffer<u32>,
    pub gate_weight: &'a DeviceBuffer<bf16>,
    pub gate_bias: &'a DeviceBuffer<bf16>,
    pub up_weight: &'a DeviceBuffer<bf16>,
    pub up_bias: &'a DeviceBuffer<bf16>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

pub struct SelectedDenseReduceLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub selected: &'a DeviceBuffer<u32>,
    pub routing: &'a DeviceBuffer<bf16>,
    pub weight: &'a DeviceBuffer<bf16>,
    pub bias: &'a DeviceBuffer<bf16>,
    pub partial: &'a mut DeviceBuffer<f32>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

#[derive(Clone, Debug)]
pub struct SelectedDenseMoe {
    gated: TypedKernel<DenseGatedKernel>,
    reduce: TypedKernel<DenseReduceKernel>,
    project: TypedKernel<DenseProjectKernel>,
    finalize: TypedKernel<DenseFinalizeKernel>,
    expert_major: Box<ExpertMajorKernels>,
    tensor_core: Box<TensorCoreKernels>,
    spec: SelectedDenseMoeSpec,
}

impl SelectedDenseMoe {
    pub fn compile(compiler: &Compiler, spec: SelectedDenseMoeSpec) -> Result<Self> {
        let spec = spec.validate()?;
        let source = cuda_kernel_file!("../../../../kernels/selected_dense_moe_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            gated: module.kernel()?,
            reduce: module.kernel()?,
            project: module.kernel()?,
            finalize: module.kernel()?,
            expert_major: Box::new(ExpertMajorKernels::new(&module)?),
            tensor_core: Box::new(TensorCoreKernels::new(&module)?),
            spec,
        })
    }

    pub fn gated(&self, stream: &Stream, launch: &mut SelectedDenseGateLaunch<'_>) -> Result<()> {
        self.validate_gated(launch)?;
        let spec = self.spec;
        let config = launch::gated(spec)?;
        Ok(self.gated.launch(
            stream,
            config,
            (
                launch.input,
                launch.selected,
                launch.gate_weight,
                launch.gate_bias,
                launch.up_weight,
                launch.up_bias,
                &mut *launch.output,
                narrow(spec.input_features)?,
                narrow(spec.output_features)?,
                narrow(spec.expert_count)?,
                narrow(spec.selected_count)?,
                spec.gate_up_layout.code(),
                u32::from(spec.gate_transposed),
                u32::from(spec.up_transposed),
                u32::from(spec.gate_bias),
                u32::from(spec.up_bias),
                spec.activation.kind,
                spec.activation.alpha,
                spec.activation.limit,
                spec.activation.up_shift,
            ),
        )?)
    }

    #[must_use]
    pub const fn spec(&self) -> SelectedDenseMoeSpec {
        self.spec
    }

    #[must_use]
    pub fn prefers_expert_major(&self) -> bool {
        self.spec.tokens > 1
            && self.spec.gate_up_layout == DenseGateUpLayout::FusedInterleaved
            && !self.spec.gate_transposed
            && !self.spec.up_transposed
            && !self.spec.down_transposed
            && self.spec.input_features.is_multiple_of(8)
            && self.spec.output_features.is_multiple_of(8)
    }
}
