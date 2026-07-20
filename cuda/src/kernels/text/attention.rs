use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{Error, Result};

cuda_export!(
    TextAttentionKernel = "libmir_cuda_text_attention_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>,
        value: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        tokens: u32, query_heads: u32, kv_heads: u32, head_dim: u32,
        scale: f32, causal: u32,
    )
);

#[derive(Clone, Copy, Debug)]
pub struct TextAttentionSpec {
    pub tokens: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub causal: bool,
}

#[derive(Clone, Debug)]
pub struct TextAttention {
    kernel: TypedKernel<TextAttentionKernel>,
    spec: TextAttentionSpec,
}

impl TextAttention {
    pub fn compile(compiler: &Compiler, spec: TextAttentionSpec) -> Result<Self> {
        if spec.tokens == 0
            || spec.query_heads == 0
            || spec.kv_heads == 0
            || !spec.query_heads.is_multiple_of(spec.kv_heads)
            || spec.head_dim == 0
            || spec.head_dim > 256
            || !spec.scale.is_finite()
            || spec.scale <= 0.0
        {
            return Err(Error::InvalidDecoderKernel("invalid text attention geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/text/attention_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key: &DeviceBuffer<bf16>,
        value: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let query_elements = self.spec.tokens * self.spec.query_heads * self.spec.head_dim;
        let kv_elements = self.spec.tokens * self.spec.kv_heads * self.spec.head_dim;
        if query.len() != query_elements
            || output.len() != query_elements
            || key.len() != kv_elements
            || value.len() != kv_elements
        {
            return Err(Error::InvalidDecoderKernel("text attention buffer geometry differs"));
        }
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(self.spec.tokens)?, u32::try_from(self.spec.query_heads)?, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                query,
                key,
                value,
                output,
                u32::try_from(self.spec.tokens)?,
                u32::try_from(self.spec.query_heads)?,
                u32::try_from(self.spec.kv_heads)?,
                u32::try_from(self.spec.head_dim)?,
                self.spec.scale,
                u32::from(self.spec.causal),
            ),
        )?)
    }
}
