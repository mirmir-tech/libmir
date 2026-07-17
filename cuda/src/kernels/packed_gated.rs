use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::GatedActivation;
use crate::{Error, Result};

cuda_export!(
    PackedGatedKernel = "libmir_cuda_packed_gated_bf16"(
        gate_input: &DeviceBuffer<bf16>, up_input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, columns: u32, elements: u32,
        separate_inputs: u32, activation: u32,
    )
);

#[derive(Clone, Debug)]
pub struct PackedGatedBf16 {
    kernel: TypedKernel<PackedGatedKernel>,
    rows: usize,
    columns: usize,
}

impl PackedGatedBf16 {
    pub(crate) fn compile(compiler: &Compiler, rows: usize, columns: usize) -> Result<Self> {
        if rows == 0 || columns == 0 {
            return Err(Error::InvalidDecoderKernel("empty packed gated geometry"));
        }
        let source = cuda_kernel_file!("../../kernels/elementwise_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, rows, columns })
    }

    pub(crate) fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        activation: GatedActivation,
    ) -> Result<()> {
        let elements = self
            .rows
            .checked_mul(self.columns)
            .ok_or(Error::InvalidDecoderKernel("packed gated size overflow"))?;
        let input_elements = elements
            .checked_mul(2)
            .ok_or(Error::InvalidDecoderKernel("packed gated size overflow"))?;
        if input.len() != input_elements || output.len() != elements {
            return Err(Error::InvalidDecoderKernel("packed gated buffers differ from geometry"));
        }
        let activation = activation_id(activation);
        Ok(self.kernel.launch(
            stream,
            LaunchConfig::for_elements(elements, 256)?,
            (
                input,
                input,
                output,
                u32::try_from(self.columns)?,
                u32::try_from(elements)?,
                0,
                activation,
            ),
        )?)
    }

    pub(crate) fn execute_separate(
        &self,
        stream: &Stream,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        activation: GatedActivation,
    ) -> Result<()> {
        let elements = self
            .rows
            .checked_mul(self.columns)
            .ok_or(Error::InvalidDecoderKernel("separate gated size overflow"))?;
        if gate.len() != elements || up.len() != elements || output.len() != elements {
            return Err(Error::InvalidDecoderKernel("separate gated buffers differ from geometry"));
        }
        Ok(self.kernel.launch(
            stream,
            LaunchConfig::for_elements(elements, 256)?,
            (
                gate,
                up,
                output,
                u32::try_from(self.columns)?,
                u32::try_from(elements)?,
                1,
                activation_id(activation),
            ),
        )?)
    }
}

const fn activation_id(activation: GatedActivation) -> u32 {
    match activation {
        GatedActivation::GeluTanh => 0,
        GatedActivation::Silu => 1,
    }
}
