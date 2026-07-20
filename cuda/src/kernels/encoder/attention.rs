use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, cuda_export,
    cuda_kernel_file, f16,
};

use crate::{Error, Result};

cuda_export!(
    AttentionKernel = "libmir_cuda_encoder_attention_f16"(
        qkv: &DeviceBuffer<f16>, output: &mut DeviceBuffer<f16>, tokens: u32,
        heads: u32, head_dim: u32, scale: f32, theta: f32, ntk_factor: f32,
    )
);

#[derive(Clone, Copy, Debug)]
pub struct EncoderAttentionSpec {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    pub ntk_factor: f32,
}

#[derive(Clone, Debug)]
pub struct EncoderAttentionF16 {
    kernel: TypedKernel<AttentionKernel>,
    spec: EncoderAttentionSpec,
}

impl EncoderAttentionF16 {
    pub fn compile(compiler: &Compiler, spec: EncoderAttentionSpec) -> Result<Self> {
        if spec.tokens == 0
            || spec.heads == 0
            || spec.head_dim == 0
            || spec.head_dim > 256
            || !spec.head_dim.is_multiple_of(2)
            || !spec.theta.is_finite()
            || spec.theta <= 0.0
            || !spec.ntk_factor.is_finite()
            || spec.ntk_factor <= 0.0
        {
            return Err(Error::InvalidDecoderKernel("invalid encoder attention geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/encoder/attention_f16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        qkv: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<f16>,
    ) -> Result<()> {
        let hidden = self.spec.heads * self.spec.head_dim;
        if qkv.len() != self.spec.tokens * hidden * 3 || output.len() != self.spec.tokens * hidden {
            return Err(Error::InvalidDecoderKernel("encoder attention buffer geometry differs"));
        }
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(self.spec.tokens)?, u32::try_from(self.spec.heads)?, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                qkv,
                output,
                u32::try_from(self.spec.tokens)?,
                u32::try_from(self.spec.heads)?,
                u32::try_from(self.spec.head_dim)?,
                self.spec.head_dim.to_string().parse::<f32>()?.sqrt().recip(),
                self.spec.theta,
                self.spec.ntk_factor,
            ),
        )?)
    }
}
