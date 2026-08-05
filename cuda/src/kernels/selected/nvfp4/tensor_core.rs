use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{NvFp4BankView, SelectedNvFp4Spec, activation, batch, validate, validate_bank};
use crate::{
    Result,
    kernels::geometry::{product, require},
};

cuda_export!(GatedKernel = "libmir_cuda_selected_nvfp4_tensor_core_gated_bf16"(
    input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    gate_weight: &DeviceBuffer<u8>, gate_scales: &DeviceBuffer<u8>,
    gate_global: &DeviceBuffer<f32>, up_weight: &DeviceBuffer<u8>,
    up_scales: &DeviceBuffer<u8>, up_global: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>, input_features: u32,
    output_features: u32, selected_count: u32, routes: u32, activation: u32,
));

cuda_export!(LinearKernel = "libmir_cuda_selected_nvfp4_tensor_core_linear_bf16"(
    input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    weight: &DeviceBuffer<u8>, scales: &DeviceBuffer<u8>,
    global_scales: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>,
    input_features: u32, output_features: u32, routes: u32,
));

#[derive(Clone, Debug)]
pub struct SelectedNvFp4TensorCoreGated {
    kernel: TypedKernel<GatedKernel>,
    spec: SelectedNvFp4Spec,
}

#[derive(Clone, Debug)]
pub struct SelectedNvFp4TensorCoreLinear {
    kernel: TypedKernel<LinearKernel>,
    spec: SelectedNvFp4Spec,
}

impl SelectedNvFp4TensorCoreGated {
    pub fn compile(compiler: &Compiler, spec: SelectedNvFp4Spec) -> Result<Self> {
        validate(spec)?;
        Ok(Self {
            kernel: compile(compiler)?.kernel()?,
            spec,
        })
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
        tokens: usize,
    ) -> Result<()> {
        let routes = batch(tokens, self.spec.selected)?;
        require("Tensor Core selected input", batch(tokens, self.spec.hidden)?, input.len())?;
        require("Tensor Core selected indices", routes, selected.len())?;
        require(
            "Tensor Core selected gated output",
            product(routes, self.spec.intermediate)?,
            output.len(),
        )?;
        validate_bank(gate, self.spec.experts, self.spec.hidden, self.spec.intermediate)?;
        validate_bank(up, self.spec.experts, self.spec.hidden, self.spec.intermediate)?;
        Ok(self.kernel.launch(
            stream,
            config(self.spec.intermediate, routes)?,
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
                u32::try_from(routes)?,
                activation(self.spec.activation),
            ),
        )?)
    }
}

impl SelectedNvFp4TensorCoreLinear {
    pub fn compile(compiler: &Compiler, spec: SelectedNvFp4Spec) -> Result<Self> {
        validate(spec)?;
        Ok(Self {
            kernel: compile(compiler)?.kernel()?,
            spec,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        down: NvFp4BankView<'_>,
        output: &mut DeviceBuffer<bf16>,
        tokens: usize,
    ) -> Result<()> {
        let routes = batch(tokens, self.spec.selected)?;
        require(
            "Tensor Core selected linear input",
            product(routes, self.spec.intermediate)?,
            input.len(),
        )?;
        require("Tensor Core selected linear indices", routes, selected.len())?;
        require(
            "Tensor Core selected linear output",
            product(routes, self.spec.hidden)?,
            output.len(),
        )?;
        validate_bank(down, self.spec.experts, self.spec.intermediate, self.spec.hidden)?;
        Ok(self.kernel.launch(
            stream,
            config(self.spec.hidden, routes)?,
            (
                input,
                selected,
                down.weight,
                down.scales,
                down.globals,
                output,
                u32::try_from(self.spec.intermediate)?,
                u32::try_from(self.spec.hidden)?,
                u32::try_from(routes)?,
            ),
        )?)
    }
}

fn compile(compiler: &Compiler) -> Result<mircuda::Module> {
    let source = cuda_kernel_file!("../../../../kernels/selected_nvfp4_tensor_core_bf16.cu");
    Ok(compiler.compile(source, &CompileOptions::default())?)
}

fn config(output_features: usize, routes: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (u32::try_from(output_features.div_ceil(128))?, u32::try_from(routes)?, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
