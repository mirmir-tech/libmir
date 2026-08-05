use std::sync::Mutex;

use mircuda::{
    CompileOptions, Compiler, Context, CublasLtFp8Plan, CublasLtFp8Spec, DeviceBuffer,
    LaunchConfig, MemoryPool, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::DirectFp8Spec;
use crate::{Error, Result, kernels::geometry::narrow};

cuda_export!(StaticE4M3QuantizeF32Kernel = "libmir_cuda_static_e4m3_quantize_bf16_f32_scale"(
    input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<u8>, scales: &mut DeviceBuffer<f32>,
    input_scale: &DeviceBuffer<f32>, tokens: u32, columns: u32,
));

#[derive(Debug)]
pub struct DirectFp8CublasLtLinear {
    quantize: TypedKernel<StaticE4M3QuantizeF32Kernel>,
    plan: Mutex<CublasLtFp8Plan>,
    quantized: DeviceBuffer<u8>,
    input_scale: DeviceBuffer<f32>,
    spec: DirectFp8Spec,
}

impl DirectFp8CublasLtLinear {
    pub(crate) fn prepare(
        compiler: &Compiler,
        context: &Context,
        pool: &MemoryPool,
        stream: &Stream,
        spec: DirectFp8Spec,
    ) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/direct_fp8_quantize.cu");
        let module = compiler.compile(
            source,
            &CompileOptions {
                fast_math: false,
                ..CompileOptions::default()
            },
        )?;
        Ok(Self {
            quantize: module.kernel()?,
            plan: Mutex::new(CublasLtFp8Plan::new(
                context,
                stream,
                CublasLtFp8Spec::new(spec.tokens, spec.output_features, spec.input_features)?,
            )?),
            quantized: pool.allocate(stream, spec.input_elements()?)?,
            input_scale: pool.allocate(stream, spec.tokens)?,
            spec,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u8>,
        weight_scale: &DeviceBuffer<f32>,
        input_scale: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut quantized = self.quantized.clone();
        let mut scales = self.input_scale.clone();
        self.quantize.launch(
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
                input_scale,
                narrow(self.spec.tokens)?,
                narrow(self.spec.input_features)?,
            ),
        )?;
        self.plan
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("cuBLASLt FP8 plan lock is poisoned"))?
            .execute(stream, &quantized, weight, &scales, weight_scale, output)?;
        Ok(())
    }
}
