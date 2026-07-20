use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    SplitKernel = "libmir_cuda_gated_attention_split_bf16"(
        input: &DeviceBuffer<bf16>, query: &mut DeviceBuffer<bf16>,
        gate: &mut DeviceBuffer<bf16>, elements: u32, heads: u32, head_dim: u32,
    )
);

#[derive(Clone, Debug)]
pub struct GatedAttentionSplit {
    kernel: TypedKernel<SplitKernel>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
}

impl GatedAttentionSplit {
    pub fn compile(
        compiler: &Compiler,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if tokens == 0 || heads == 0 || head_dim == 0 {
            return Err(Error::InvalidDecoderKernel("empty gated attention split"));
        }
        let source = cuda_kernel_file!("../../kernels/gated_attention_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            tokens,
            heads,
            head_dim,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        query: &mut DeviceBuffer<bf16>,
        gate: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let width = product(self.heads, self.head_dim)?;
        let elements = product(self.tokens, width)?;
        require("gated query projection", product(elements, 2)?, input.len())?;
        require("gated query", elements, query.len())?;
        require("attention output gate", elements, gate.len())?;
        let threads = 256_usize;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                query,
                gate,
                narrow(elements)?,
                narrow(self.heads)?,
                narrow(self.head_dim)?,
            ),
        )?)
    }
}
