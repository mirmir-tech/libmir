use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    EmbeddingKernel = "libmir_cuda_embedding_bf16"(
        weight: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>, selected_start: u32, tokens: u32,
        vocab: u32, hidden: u32, scale: f32,
    )
);

#[derive(Clone, Debug)]
pub struct Embedding {
    kernel: TypedKernel<EmbeddingKernel>,
    vocab: usize,
    hidden: usize,
    scale: f32,
}

impl Embedding {
    pub fn compile(compiler: &Compiler, vocab: usize, hidden: usize, scale: f32) -> Result<Self> {
        if vocab == 0 || hidden == 0 {
            return Err(Error::InvalidDecoderKernel("invalid embedding geometry"));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::InvalidDecoderKernel("invalid embedding scale"));
        }
        let source = cuda_kernel_file!("../../kernels/embedding_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            vocab,
            hidden,
            scale,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        weight: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        selected_index: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_batch(stream, weight, selected, selected_index, 1, output)
    }

    pub fn execute_batch(
        &self,
        stream: &Stream,
        weight: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("embedding token batch is empty"));
        }
        let elements = product(self.vocab, self.hidden)?;
        require("embedding weight", elements, weight.len())?;
        let required_tokens = selected_start
            .checked_add(tokens)
            .ok_or(Error::InvalidDecoderKernel("embedding token index overflow"))?;
        require("embedding token", required_tokens, selected.len())?;
        let output_elements = product(tokens, self.hidden)?;
        require("embedding output", output_elements, output.len())?;
        let threads = 256_u32;
        let config = LaunchConfig {
            grid: (narrow(output_elements.div_ceil(usize::try_from(threads)?))?, 1, 1),
            block: (threads, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.kernel.launch(
            stream,
            config,
            (
                weight,
                output,
                selected,
                narrow(selected_start)?,
                narrow(tokens)?,
                narrow(self.vocab)?,
                narrow(self.hidden)?,
                self.scale,
            ),
        )?)
    }
}
