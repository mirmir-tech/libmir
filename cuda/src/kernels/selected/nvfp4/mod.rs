use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::GatedActivation;
use crate::{
    Error, Result,
    kernels::geometry::{product, require},
};

mod tensor_core;
mod tiled;

pub use tensor_core::{SelectedNvFp4TensorCoreGated, SelectedNvFp4TensorCoreLinear};
pub use tiled::{SelectedNvFp4TiledGated, SelectedNvFp4TiledReduce, SelectedNvFp4TiledRows};

cuda_export!(
    GatedKernel = "libmir_cuda_selected_nvfp4_gated_bf16"(
        input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
        gate_weight: &DeviceBuffer<u8>, gate_scales: &DeviceBuffer<u8>,
        gate_global: &DeviceBuffer<f32>, up_weight: &DeviceBuffer<u8>,
        up_scales: &DeviceBuffer<u8>, up_global: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>, input_features: u32,
        output_features: u32, selected_count: u32, tokens: u32, activation: u32,
    )
);

cuda_export!(
    ReduceKernel = "libmir_cuda_selected_nvfp4_reduce_bf16"(
        input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>, global_scales: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>, input_features: u32,
        output_features: u32, selected_count: u32, tokens: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectedNvFp4Spec {
    pub experts: usize,
    pub selected: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub activation: GatedActivation,
}

#[derive(Clone, Debug)]
pub struct SelectedNvFp4Gated {
    kernel: TypedKernel<GatedKernel>,
    spec: SelectedNvFp4Spec,
}

#[derive(Clone, Debug)]
pub struct SelectedNvFp4Reduce {
    kernel: TypedKernel<ReduceKernel>,
    spec: SelectedNvFp4Spec,
}

impl SelectedNvFp4Gated {
    pub fn compile(compiler: &Compiler, spec: SelectedNvFp4Spec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../../kernels/selected_nvfp4_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        gate: NvFp4BankView<'_>,
        up: NvFp4BankView<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_batch(stream, input, selected, gate, up, output, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_batch(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        gate: NvFp4BankView<'_>,
        up: NvFp4BankView<'_>,
        output: &mut DeviceBuffer<bf16>,
        tokens: usize,
    ) -> Result<()> {
        let selected_elements = batch(tokens, self.spec.selected)?;
        require("selected NVFP4 input", batch(tokens, self.spec.hidden)?, input.len())?;
        require("selected NVFP4 indices", selected_elements, selected.len())?;
        require(
            "selected NVFP4 gated output",
            product(selected_elements, self.spec.intermediate)?,
            output.len(),
        )?;
        validate_bank(gate, self.spec.experts, self.spec.hidden, self.spec.intermediate)?;
        validate_bank(up, self.spec.experts, self.spec.hidden, self.spec.intermediate)?;
        let config = LaunchConfig {
            grid: (
                u32::try_from(self.spec.intermediate.div_ceil(8))?,
                u32::try_from(self.spec.selected)?,
                u32::try_from(tokens)?,
            ),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.kernel.launch(
            stream,
            config,
            (
                input,
                selected,
                gate.weight,
                gate.scales,
                gate.globals,
                up.weight,
                up.scales,
                up.globals,
                output,
                u32::try_from(self.spec.hidden)?,
                u32::try_from(self.spec.intermediate)?,
                u32::try_from(self.spec.selected)?,
                u32::try_from(tokens)?,
                activation(self.spec.activation),
            ),
        )?)
    }
}

impl SelectedNvFp4Reduce {
    pub fn compile(compiler: &Compiler, spec: SelectedNvFp4Spec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../../kernels/selected_nvfp4_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        down: NvFp4BankView<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_batch(stream, input, selected, routing, down, output, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_batch(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        down: NvFp4BankView<'_>,
        output: &mut DeviceBuffer<bf16>,
        tokens: usize,
    ) -> Result<()> {
        let selected_elements = batch(tokens, self.spec.selected)?;
        require(
            "selected NVFP4 intermediate",
            product(selected_elements, self.spec.intermediate)?,
            input.len(),
        )?;
        require("selected NVFP4 indices", selected_elements, selected.len())?;
        require("selected NVFP4 routing", selected_elements, routing.len())?;
        require("selected NVFP4 output", batch(tokens, self.spec.hidden)?, output.len())?;
        validate_bank(down, self.spec.experts, self.spec.intermediate, self.spec.hidden)?;
        let config = LaunchConfig {
            grid: (u32::try_from(self.spec.hidden.div_ceil(8))?, u32::try_from(tokens)?, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.kernel.launch(
            stream,
            config,
            (
                input,
                selected,
                routing,
                down.weight,
                down.scales,
                down.globals,
                output,
                u32::try_from(self.spec.intermediate)?,
                u32::try_from(self.spec.hidden)?,
                u32::try_from(self.spec.selected)?,
                u32::try_from(tokens)?,
            ),
        )?)
    }
}

#[derive(Clone, Copy)]
pub struct NvFp4BankView<'a> {
    pub weight: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub globals: &'a DeviceBuffer<f32>,
    pub input_globals: &'a DeviceBuffer<f32>,
    pub combined: &'a DeviceBuffer<f32>,
}

pub(super) fn validate_bank(
    bank: NvFp4BankView<'_>,
    experts: usize,
    input: usize,
    output: usize,
) -> Result<()> {
    require("NVFP4 bank weight", experts * output * input / 2, bank.weight.len())?;
    require("NVFP4 bank scales", experts * output * input / 16, bank.scales.len())?;
    require("NVFP4 bank globals", experts, bank.globals.len())?;
    require("NVFP4 bank input globals", experts, bank.input_globals.len())?;
    require("NVFP4 bank combined scales", experts, bank.combined.len())
}

pub(super) fn validate(spec: SelectedNvFp4Spec) -> Result<()> {
    if spec.experts == 0
        || spec.selected == 0
        || spec.selected > spec.experts
        || spec.hidden == 0
        || spec.intermediate == 0
        || !spec.hidden.is_multiple_of(16)
        || !spec.intermediate.is_multiple_of(16)
    {
        Err(Error::InvalidNvFp4("invalid selected expert geometry"))
    } else {
        Ok(())
    }
}

pub(super) fn batch(tokens: usize, width: usize) -> Result<usize> {
    if tokens == 0 {
        Err(Error::InvalidNvFp4("selected expert batch is empty"))
    } else {
        product(tokens, width)
    }
}

pub(super) const fn activation(value: GatedActivation) -> u32 {
    match value {
        GatedActivation::GeluTanh => 0,
        GatedActivation::Silu => 1,
    }
}
