use std::mem::size_of;

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{super::geometry::narrow, DirectFp8Spec, DirectFp8TensorCoreLinear};
use crate::{Error, Result};

cuda_export!(DynamicE4M3NormGateKernel = "libmir_cuda_dynamic_e4m3_norm_gate_bf16"(
    input: &DeviceBuffer<bf16>, gate: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<u8>, scales: &mut DeviceBuffer<f32>, tokens: u32, columns: u32,
    value_heads: u32, gate_stride: u32, gate_offset: u32, epsilon: f32, weight_shift: f32,
));
#[cfg(test)]
cuda_export!(ReferenceDynamicE4M3Kernel = "libmir_cuda_dynamic_e4m3_quantize_bf16"(
    input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<u8>, scales: &mut DeviceBuffer<f32>,
    tokens: u32, columns: u32,
));

#[derive(Debug)]
pub struct DirectFp8NormGate {
    kernel: TypedKernel<DynamicE4M3NormGateKernel>,
    #[cfg(test)]
    reference: TypedKernel<ReferenceDynamicE4M3Kernel>,
}

impl DirectFp8NormGate {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/direct_fp8_quantize.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            #[cfg(test)]
            reference: module.kernel()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        spec: DirectFp8Spec,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<f32>,
        value_heads: usize,
        gate_stride: usize,
        gate_offset: usize,
        epsilon: f32,
        weight_shift: f32,
    ) -> Result<()> {
        if value_heads == 0
            || spec.input_features != value_heads * 128
            || input.len() < spec.input_elements()?
            || weight.len() < 128
            || output.len() < spec.input_elements()?
            || scales.len() < spec.tokens
        {
            return Err(Error::InvalidDecoderKernel("invalid fused FP8 norm-gate geometry"));
        }
        let gate_end = spec
            .tokens
            .saturating_sub(1)
            .checked_mul(gate_stride)
            .and_then(|offset| offset.checked_add(gate_offset + spec.input_features))
            .ok_or(Error::InvalidDecoderKernel("fused FP8 norm-gate overflow"))?;
        if gate.len() < gate_end {
            return Err(Error::InvalidDecoderKernel("fused FP8 norm-gate input is too small"));
        }
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(spec.tokens)?, 1, 1),
                block: (384, 1, 1),
                shared_memory_bytes: narrow(spec.input_features * size_of::<bf16>())?,
            },
            (
                input,
                gate,
                weight,
                output,
                scales,
                narrow(spec.tokens)?,
                narrow(spec.input_features)?,
                narrow(value_heads)?,
                narrow(gate_stride)?,
                narrow(gate_offset)?,
                epsilon,
                weight_shift,
            ),
        )?)
    }

    #[cfg(test)]
    pub(crate) fn quantize_reference(
        &self,
        stream: &Stream,
        spec: DirectFp8Spec,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        Ok(self.reference.launch(
            stream,
            LaunchConfig {
                grid: (narrow(spec.tokens)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, output, scales, narrow(spec.tokens)?, narrow(spec.input_features)?),
        )?)
    }
}

impl DirectFp8TensorCoreLinear {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_norm_gate_f32_scales(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        norm: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<f32>,
        bias: Option<&DeviceBuffer<bf16>>,
        epsilon: f32,
        weight_shift: f32,
        value_heads: usize,
        gate_stride: usize,
        gate_offset: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut quantized = self.quantized.clone();
        let mut input_scales = self.input_scales.clone();
        self.norm_gate.execute(
            stream,
            self.spec,
            input,
            gate,
            norm,
            &mut quantized,
            &mut input_scales,
            value_heads,
            gate_stride,
            gate_offset,
            epsilon,
            weight_shift,
        )?;
        Ok(self.plan.execute_f32_scales(
            stream,
            &self.quantized,
            weight,
            &self.input_scales,
            weight_scales,
            bias,
            output,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_norm_gate_bf16_scales(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        norm: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<bf16>,
        bias: Option<&DeviceBuffer<bf16>>,
        epsilon: f32,
        weight_shift: f32,
        value_heads: usize,
        gate_stride: usize,
        gate_offset: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut quantized = self.quantized.clone();
        let mut input_scales = self.input_scales.clone();
        self.norm_gate.execute(
            stream,
            self.spec,
            input,
            gate,
            norm,
            &mut quantized,
            &mut input_scales,
            value_heads,
            gate_stride,
            gate_offset,
            epsilon,
            weight_shift,
        )?;
        Ok(self.plan.execute_bf16_scales(
            stream,
            &self.quantized,
            weight,
            &self.input_scales,
            weight_scales,
            bias,
            output,
        )?)
    }
}
