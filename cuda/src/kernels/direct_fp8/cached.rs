use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{DirectFp8Activation, DirectFp8Format, DirectFp8Scales, DirectFp8Spec};
use crate::{
    Result,
    kernels::geometry::{narrow, require},
};

cuda_export!(CachedF32ScaleKernel = "libmir_cuda_direct_fp8_bf16_linear_f32_scale_cached"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u8>, scales: &DeviceBuffer<f32>,
    input_scale: &DeviceBuffer<f32>, bias: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>, tokens: u32, rows: u32, columns: u32,
    scale_rows: u32, scale_columns: u32, scale_row_size: u32, scale_group_size: u32,
    inverse_scale: u32, has_bias: u32, activation_mode: u32, e5m2: u32,
    cache_input: u32,
));

#[derive(Clone, Debug)]
pub struct DirectFp8CachedLinear {
    kernel: TypedKernel<CachedF32ScaleKernel>,
    spec: DirectFp8Spec,
}

impl DirectFp8CachedLinear {
    pub fn compile(compiler: &Compiler, spec: DirectFp8Spec) -> Result<Self> {
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/direct_fp8.cu"),
            &CompileOptions {
                fast_math: false,
                ..CompileOptions::default()
            },
        )?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        scales: DirectFp8Scales<'_, f32>,
        bias: Option<&DeviceBuffer<bf16>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("cached direct FP8 input", self.spec.input_elements()?, input.len())?;
        require("cached direct FP8 weight", self.spec.weight_elements()?, weight.len())?;
        require("cached direct FP8 scales", self.spec.scale_elements()?, scales.weight.len())?;
        require("cached direct FP8 activation scale", 1, scales.activation.len())?;
        require("cached direct FP8 output", self.spec.output_elements()?, output.len())?;
        let (scale_rows, scale_columns, scale_row_size, scale_group_size) =
            self.spec.scale_geometry()?;
        let cache_input = self.spec.input_features <= 8_192;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(self.spec.output_features.div_ceil(64))?,
                    narrow(self.spec.tokens)?,
                    1,
                ),
                block: (256, 1, 1),
                shared_memory_bytes: if cache_input {
                    narrow(self.spec.input_features.checked_mul(size_of::<f32>()).ok_or(
                        crate::Error::InvalidDecoderKernel("cached direct FP8 input overflow"),
                    )?)?
                } else {
                    0
                },
            },
            (
                input,
                weight,
                scales.weight,
                scales.activation,
                bias.unwrap_or(input),
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.output_features)?,
                narrow(self.spec.input_features)?,
                narrow(scale_rows)?,
                narrow(scale_columns)?,
                narrow(scale_row_size)?,
                narrow(scale_group_size)?,
                u32::from(self.spec.inverse_scale),
                u32::from(bias.is_some()),
                activation_mode(self.spec.activation),
                u32::from(self.spec.format == DirectFp8Format::E5M2),
                u32::from(cache_input),
            ),
        )?)
    }
}

const fn activation_mode(activation: DirectFp8Activation) -> u32 {
    match activation {
        DirectFp8Activation::Bf16 => 0,
        DirectFp8Activation::DynamicE4M3Token => 1,
        DirectFp8Activation::StaticE4M3Tensor => 2,
    }
}
