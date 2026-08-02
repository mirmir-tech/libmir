use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(MxFp4EmbeddingKernel = "libmir_cuda_mxfp4_embedding_bf16"(
    weight: &DeviceBuffer<u8>, scales: &DeviceBuffer<u8>, selected: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, selected_start: u32, tokens: u32, vocab: u32, hidden: u32,
    output_scale: f32,
));

#[derive(Clone, Copy, Debug)]
pub struct MxFp4EmbeddingSpec {
    pub vocab: usize,
    pub hidden: usize,
    pub output_scale: f32,
}

impl MxFp4EmbeddingSpec {
    pub fn new(vocab: usize, hidden: usize, output_scale: f32) -> Result<Self> {
        if vocab == 0 || hidden == 0 || !hidden.is_multiple_of(32) || !output_scale.is_finite() {
            return Err(Error::InvalidDecoderKernel("invalid MXFP4 embedding geometry"));
        }
        Ok(Self { vocab, hidden, output_scale })
    }
}

#[derive(Clone, Debug)]
pub struct MxFp4Embedding {
    kernel: TypedKernel<MxFp4EmbeddingKernel>,
    spec: MxFp4EmbeddingSpec,
}

pub struct MxFp4EmbeddingOperands<'a> {
    pub weight: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub selected: &'a DeviceBuffer<u32>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl MxFp4Embedding {
    pub fn compile(compiler: &Compiler, spec: MxFp4EmbeddingSpec) -> Result<Self> {
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/mxfp4_linear.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        operands: &mut MxFp4EmbeddingOperands<'_>,
        selected_start: usize,
        tokens: usize,
    ) -> Result<()> {
        let selected_end = selected_start
            .checked_add(tokens)
            .ok_or(Error::InvalidDecoderKernel("MXFP4 selected range overflow"))?;
        require("MXFP4 selected tokens", selected_end, operands.selected.len())?;
        require(
            "MXFP4 embedding weight",
            product(self.spec.vocab, self.spec.hidden / 2)?,
            operands.weight.len(),
        )?;
        require(
            "MXFP4 embedding scales",
            product(self.spec.vocab, self.spec.hidden / 32)?,
            operands.scales.len(),
        )?;
        require(
            "MXFP4 embedding output",
            product(tokens, self.spec.hidden)?,
            operands.output.len(),
        )?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.hidden.div_ceil(256))?, narrow(tokens)?, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                operands.weight,
                operands.scales,
                operands.selected,
                &mut *operands.output,
                narrow(selected_start)?,
                narrow(tokens)?,
                narrow(self.spec.vocab)?,
                narrow(self.spec.hidden)?,
                self.spec.output_scale,
            ),
        )?)
    }
}
