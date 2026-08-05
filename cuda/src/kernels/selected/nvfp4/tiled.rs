use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{NvFp4BankView, SelectedNvFp4Spec, activation, batch, validate, validate_bank};
use crate::{
    Result,
    kernels::geometry::{product, require},
};

macro_rules! gated_export {
    ($type:ident, $symbol:literal) => {
        cuda_export!(
            $type = $symbol(
                input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
                gate_weight: &DeviceBuffer<u8>, gate_scales: &DeviceBuffer<u8>,
                gate_global: &DeviceBuffer<f32>, up_weight: &DeviceBuffer<u8>,
                up_scales: &DeviceBuffer<u8>, up_global: &DeviceBuffer<f32>,
                output: &mut DeviceBuffer<bf16>, input_features: u32,
                output_features: u32, selected_count: u32, tokens: u32, activation: u32,
            )
        );
    };
}

macro_rules! reduce_export {
    ($type:ident, $symbol:literal) => {
        cuda_export!(
            $type = $symbol(
                input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
                routing: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>,
                scales: &DeviceBuffer<u8>, global_scales: &DeviceBuffer<f32>,
                output: &mut DeviceBuffer<bf16>, input_features: u32,
                output_features: u32, selected_count: u32, tokens: u32,
            )
        );
    };
}

gated_export!(Gated2, "libmir_cuda_selected_nvfp4_gated_tiled2_bf16");
gated_export!(Gated4, "libmir_cuda_selected_nvfp4_gated_tiled4_bf16");
gated_export!(Gated8, "libmir_cuda_selected_nvfp4_gated_tiled8_bf16");
reduce_export!(Reduce2, "libmir_cuda_selected_nvfp4_reduce_tiled2_bf16");
reduce_export!(Reduce4, "libmir_cuda_selected_nvfp4_reduce_tiled4_bf16");
reduce_export!(Reduce8, "libmir_cuda_selected_nvfp4_reduce_tiled8_bf16");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectedNvFp4TiledRows {
    Two,
    Four,
    Eight,
}

impl SelectedNvFp4TiledRows {
    const fn get(self) -> usize {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

#[derive(Clone, Debug)]
enum Gated {
    Two(TypedKernel<Gated2>),
    Four(TypedKernel<Gated4>),
    Eight(TypedKernel<Gated8>),
}

#[derive(Clone, Debug)]
enum Reduce {
    Two(TypedKernel<Reduce2>),
    Four(TypedKernel<Reduce4>),
    Eight(TypedKernel<Reduce8>),
}

#[derive(Clone, Debug)]
pub struct SelectedNvFp4TiledGated {
    kernel: Gated,
    spec: SelectedNvFp4Spec,
    rows: SelectedNvFp4TiledRows,
}

#[derive(Clone, Debug)]
pub struct SelectedNvFp4TiledReduce {
    kernel: Reduce,
    spec: SelectedNvFp4Spec,
    rows: SelectedNvFp4TiledRows,
}

impl SelectedNvFp4TiledGated {
    pub fn compile(
        compiler: &Compiler,
        spec: SelectedNvFp4Spec,
        rows: SelectedNvFp4TiledRows,
    ) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../../kernels/selected_nvfp4_tiled_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        let kernel = match rows {
            SelectedNvFp4TiledRows::Two => Gated::Two(module.kernel()?),
            SelectedNvFp4TiledRows::Four => Gated::Four(module.kernel()?),
            SelectedNvFp4TiledRows::Eight => Gated::Eight(module.kernel()?),
        };
        Ok(Self { kernel, spec, rows })
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
        require("tiled selected NVFP4 input", batch(tokens, self.spec.hidden)?, input.len())?;
        require("tiled selected NVFP4 indices", selected_elements, selected.len())?;
        require(
            "tiled selected NVFP4 gated output",
            product(selected_elements, self.spec.intermediate)?,
            output.len(),
        )?;
        validate_bank(gate, self.spec.experts, self.spec.hidden, self.spec.intermediate)?;
        validate_bank(up, self.spec.experts, self.spec.hidden, self.spec.intermediate)?;
        let config = config(self.spec.intermediate, self.spec.selected, tokens, self.rows)?;
        let args = (
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
        );
        match &self.kernel {
            Gated::Two(kernel) => kernel.launch(stream, config, args),
            Gated::Four(kernel) => kernel.launch(stream, config, args),
            Gated::Eight(kernel) => kernel.launch(stream, config, args),
        }?;
        Ok(())
    }
}

impl SelectedNvFp4TiledReduce {
    pub fn compile(
        compiler: &Compiler,
        spec: SelectedNvFp4Spec,
        rows: SelectedNvFp4TiledRows,
    ) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../../kernels/selected_nvfp4_tiled_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        let kernel = match rows {
            SelectedNvFp4TiledRows::Two => Reduce::Two(module.kernel()?),
            SelectedNvFp4TiledRows::Four => Reduce::Four(module.kernel()?),
            SelectedNvFp4TiledRows::Eight => Reduce::Eight(module.kernel()?),
        };
        Ok(Self { kernel, spec, rows })
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
            "tiled selected NVFP4 intermediate",
            product(selected_elements, self.spec.intermediate)?,
            input.len(),
        )?;
        require("tiled selected NVFP4 indices", selected_elements, selected.len())?;
        require("tiled selected NVFP4 routing", selected_elements, routing.len())?;
        require("tiled selected NVFP4 output", batch(tokens, self.spec.hidden)?, output.len())?;
        validate_bank(down, self.spec.experts, self.spec.intermediate, self.spec.hidden)?;
        let config = config(self.spec.hidden, tokens, 1, self.rows)?;
        let args = (
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
        );
        match &self.kernel {
            Reduce::Two(kernel) => kernel.launch(stream, config, args),
            Reduce::Four(kernel) => kernel.launch(stream, config, args),
            Reduce::Eight(kernel) => kernel.launch(stream, config, args),
        }?;
        Ok(())
    }
}

fn config(
    output: usize,
    grid_y: usize,
    grid_z: usize,
    rows: SelectedNvFp4TiledRows,
) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (
            u32::try_from(output.div_ceil(8 * rows.get()))?,
            u32::try_from(grid_y)?,
            u32::try_from(grid_z)?,
        ),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
