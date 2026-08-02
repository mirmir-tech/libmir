use mircuda::{
    CompileOptions, Compiler, Context, DeviceBuffer, LaunchConfig, MemoryPool, ScaledFp8Plan,
    ScaledFp8Scale, ScaledFp8Spec, ScaledFp8WeightScale, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{DirectFp8Activation, DirectFp8Scale, DirectFp8Spec};
use crate::{Error, Result, kernels::geometry::narrow};

cuda_export!(DynamicE4M3QuantizeKernel = "libmir_cuda_dynamic_e4m3_quantize_bf16"(
    input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<u8>, scales: &mut DeviceBuffer<f32>,
    tokens: u32, columns: u32,
));
cuda_export!(StaticE4M3QuantizeF32Kernel = "libmir_cuda_static_e4m3_quantize_bf16_f32_scale"(
    input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<u8>, scales: &mut DeviceBuffer<f32>,
    input_scale: &DeviceBuffer<f32>, tokens: u32, columns: u32,
));
cuda_export!(StaticE4M3QuantizeBf16Kernel = "libmir_cuda_static_e4m3_quantize_bf16_bf16_scale"(
    input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<u8>, scales: &mut DeviceBuffer<f32>,
    input_scale: &DeviceBuffer<bf16>, tokens: u32, columns: u32,
));

#[derive(Debug)]
pub struct DirectFp8TensorCoreLinear {
    dynamic_quantize: TypedKernel<DynamicE4M3QuantizeKernel>,
    static_quantize_f32: TypedKernel<StaticE4M3QuantizeF32Kernel>,
    static_quantize_bf16: TypedKernel<StaticE4M3QuantizeBf16Kernel>,
    plan: ScaledFp8Plan,
    quantized: DeviceBuffer<u8>,
    input_scales: DeviceBuffer<f32>,
    spec: DirectFp8Spec,
}

impl DirectFp8TensorCoreLinear {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        compiler: &Compiler,
        context: &Context,
        pool: &MemoryPool,
        stream: &Stream,
        spec: DirectFp8Spec,
        scale: ScaledFp8Scale,
        has_bias: bool,
    ) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/direct_fp8_quantize.cu");
        let module = compiler.compile(
            source,
            &CompileOptions {
                fast_math: false,
                ..CompileOptions::default()
            },
        )?;
        let plan = ScaledFp8Plan::new(
            context,
            stream,
            ScaledFp8Spec::new(
                spec.tokens,
                spec.input_features,
                spec.output_features,
                scale,
                match spec.scale {
                    DirectFp8Scale::Tensor => ScaledFp8WeightScale::Tensor,
                    DirectFp8Scale::OutputChannel => ScaledFp8WeightScale::OutputChannel,
                    DirectFp8Scale::BlockGrid { .. } => {
                        return Err(Error::InvalidExecutionPlan(
                            "direct FP8 Tensor Core does not accept a block scale grid",
                        ));
                    },
                },
                has_bias,
            )?,
        )?;
        Ok(Self {
            dynamic_quantize: module.kernel()?,
            static_quantize_f32: module.kernel()?,
            static_quantize_bf16: module.kernel()?,
            plan,
            quantized: pool.allocate::<u8>(stream, spec.input_elements()?)?,
            input_scales: pool.allocate::<f32>(stream, spec.tokens)?,
            spec,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_f32_scales(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<f32>,
        input_scale: Option<&DeviceBuffer<f32>>,
        bias: Option<&DeviceBuffer<bf16>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.quantize_f32(stream, input, input_scale)?;
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
    pub(crate) fn execute_bf16_scales(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<bf16>,
        input_scale: Option<&DeviceBuffer<bf16>>,
        bias: Option<&DeviceBuffer<bf16>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.quantize_bf16(stream, input, input_scale)?;
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

    fn quantize_f32(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        scale: Option<&DeviceBuffer<f32>>,
    ) -> Result<()> {
        match self.spec.activation {
            DirectFp8Activation::DynamicE4M3Token => self.quantize_dynamic(stream, input),
            DirectFp8Activation::StaticE4M3Tensor => self.quantize_static_f32(
                stream,
                input,
                scale.ok_or(Error::InvalidExecutionPlan(
                    "static direct FP8 Tensor Core input scale is missing",
                ))?,
            ),
            DirectFp8Activation::Bf16 => {
                Err(Error::InvalidExecutionPlan("direct FP8 Tensor Core requires E4M3 activations"))
            },
        }
    }

    fn quantize_bf16(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        scale: Option<&DeviceBuffer<bf16>>,
    ) -> Result<()> {
        match self.spec.activation {
            DirectFp8Activation::DynamicE4M3Token => self.quantize_dynamic(stream, input),
            DirectFp8Activation::StaticE4M3Tensor => self.quantize_static_bf16(
                stream,
                input,
                scale.ok_or(Error::InvalidExecutionPlan(
                    "static direct FP8 Tensor Core input scale is missing",
                ))?,
            ),
            DirectFp8Activation::Bf16 => {
                Err(Error::InvalidExecutionPlan("direct FP8 Tensor Core requires E4M3 activations"))
            },
        }
    }

    fn quantize_dynamic(&self, stream: &Stream, input: &DeviceBuffer<bf16>) -> Result<()> {
        let mut quantized = self.quantized.clone();
        let mut scales = self.input_scales.clone();
        Ok(self.dynamic_quantize.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.tokens)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                &mut quantized,
                &mut scales,
                narrow(self.spec.tokens)?,
                narrow(self.spec.input_features)?,
            ),
        )?)
    }

    fn quantize_static_f32(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        scale: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let mut quantized = self.quantized.clone();
        let mut scales = self.input_scales.clone();
        Ok(self.static_quantize_f32.launch(
            stream,
            self.launch()?,
            (input, &mut quantized, &mut scales, scale, self.tokens()?, self.columns()?),
        )?)
    }

    fn quantize_static_bf16(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        scale: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut quantized = self.quantized.clone();
        let mut scales = self.input_scales.clone();
        Ok(self.static_quantize_bf16.launch(
            stream,
            self.launch()?,
            (input, &mut quantized, &mut scales, scale, self.tokens()?, self.columns()?),
        )?)
    }

    fn launch(&self) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            grid: (self.tokens()?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        })
    }

    fn tokens(&self) -> Result<u32> {
        narrow(self.spec.tokens)
    }

    fn columns(&self) -> Result<u32> {
        narrow(self.spec.input_features)
    }
}
