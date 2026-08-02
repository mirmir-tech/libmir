use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::MxFp4GatheredSpec;
use crate::{
    Result,
    kernels::geometry::{narrow, require},
};

cuda_export!(MxFp4GatheredKernel = "libmir_cuda_mxfp4_bf16_gathered_linear"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>, scales: &DeviceBuffer<u8>,
    bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>,
    assignments: u32, matrices: u32, rows: u32, columns: u32,
    selections_per_input: u32, has_bias: u32,
));

#[derive(Clone, Debug)]
/// Direct gathered execution over an OCP MXFP4 matrix bank.
pub struct MxFp4GatheredLinear {
    kernel: TypedKernel<MxFp4GatheredKernel>,
    spec: MxFp4GatheredSpec,
    warps_per_block: usize,
}

pub struct MxFp4GatheredOperands<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub weight: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub bias: Option<&'a DeviceBuffer<bf16>>,
    pub selected: &'a DeviceBuffer<u32>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl MxFp4GatheredLinear {
    pub fn compile(compiler: &Compiler, spec: MxFp4GatheredSpec) -> Result<Self> {
        Self::compile_warps(compiler, spec, 8)
    }

    pub fn compile_warps(
        compiler: &Compiler,
        spec: MxFp4GatheredSpec,
        warps_per_block: usize,
    ) -> Result<Self> {
        if !matches!(warps_per_block, 1 | 2 | 4 | 8) {
            return Err(crate::Error::InvalidDecoderKernel("invalid gathered MXFP4 warp geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/mxfp4_linear.cu"),
            &CompileOptions {
                fast_math: false,
                ..CompileOptions::default()
            },
        )?;
        Ok(Self {
            kernel: module.kernel()?,
            spec,
            warps_per_block,
        })
    }

    pub fn execute(&self, stream: &Stream, operands: &mut MxFp4GatheredOperands<'_>) -> Result<()> {
        let projection = self.spec.projection()?;
        require("gathered MXFP4 input", projection.input_elements()?, operands.input.len())?;
        require("gathered MXFP4 weight", self.spec.weight_elements()?, operands.weight.len())?;
        require("gathered MXFP4 scales", self.spec.scale_elements()?, operands.scales.len())?;
        require("gathered MXFP4 indices", self.spec.assignments, operands.selected.len())?;
        if let Some(bias) = operands.bias {
            require("gathered MXFP4 bias", self.spec.bias_elements()?, bias.len())?;
        }
        require("gathered MXFP4 output", self.spec.output_elements()?, operands.output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(self.spec.output_features.div_ceil(self.warps_per_block))?,
                    narrow(self.spec.assignments)?,
                    1,
                ),
                block: (narrow(self.warps_per_block * 32)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                operands.input,
                operands.weight,
                operands.scales,
                operands.bias.unwrap_or(operands.input),
                operands.selected,
                &mut *operands.output,
                narrow(self.spec.assignments)?,
                narrow(self.spec.matrices)?,
                narrow(self.spec.output_features)?,
                narrow(self.spec.input_features)?,
                narrow(self.spec.selections_per_input)?,
                u32::from(operands.bias.is_some()),
            ),
        )?)
    }
}
