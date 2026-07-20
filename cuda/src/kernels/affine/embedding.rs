use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    Int4Kernel = "libmir_cuda_affine_embedding_bf16_int4"(
        weight: &DeviceBuffer<u32>, scales: &DeviceBuffer<bf16>, biases: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>, selected_start: u32,
        tokens: u32, vocab: u32, hidden: u32, group_size: u32, output_scale: f32,
    )
);
cuda_export!(
    Int8Kernel = "libmir_cuda_affine_embedding_bf16_int8"(
        weight: &DeviceBuffer<u32>, scales: &DeviceBuffer<bf16>, biases: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>, selected_start: u32,
        tokens: u32, vocab: u32, hidden: u32, group_size: u32, output_scale: f32,
    )
);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineEmbeddingSpec {
    pub vocab: usize,
    pub hidden: usize,
    pub group_size: usize,
    pub bits: usize,
    pub output_scale: f32,
}

#[derive(Clone, Debug)]
pub struct AffineEmbedding {
    kernel: Kernel,
    spec: AffineEmbeddingSpec,
}

#[derive(Clone, Debug)]
enum Kernel {
    Int4(TypedKernel<Int4Kernel>),
    Int8(TypedKernel<Int8Kernel>),
}

impl AffineEmbedding {
    pub fn compile(compiler: &Compiler, spec: AffineEmbeddingSpec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../kernels/affine_embedding_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        let kernel = match spec.bits {
            4 => Kernel::Int4(module.kernel()?),
            8 => Kernel::Int8(module.kernel()?),
            _ => return Err(Error::InvalidQuantizedGemv("unsupported embedding precision")),
        };
        Ok(Self { kernel, spec })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<bf16>,
        biases: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if tokens == 0 {
            return Err(Error::InvalidQuantizedGemv("embedding token batch is empty"));
        }
        let values_per_word = 32 / self.spec.bits;
        require(
            "affine embedding weight",
            product(self.spec.vocab, self.spec.hidden / values_per_word)?,
            weight.len(),
        )?;
        let groups = product(self.spec.vocab, self.spec.hidden / self.spec.group_size)?;
        require("affine embedding scales", groups, scales.len())?;
        require("affine embedding biases", groups, biases.len())?;
        require(
            "affine embedding selected",
            selected_start
                .checked_add(tokens)
                .ok_or(Error::InvalidQuantizedGemv("embedding token index overflow"))?,
            selected.len(),
        )?;
        let elements = product(tokens, self.spec.hidden)?;
        require("affine embedding output", elements, output.len())?;
        let launch = LaunchConfig {
            grid: (narrow(elements.div_ceil(256))?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        let arguments = (
            weight,
            scales,
            biases,
            selected,
            output,
            narrow(selected_start)?,
            narrow(tokens)?,
            narrow(self.spec.vocab)?,
            narrow(self.spec.hidden)?,
            narrow(self.spec.group_size)?,
            self.spec.output_scale,
        );
        Ok(match &self.kernel {
            Kernel::Int4(kernel) => kernel.launch(stream, launch, arguments),
            Kernel::Int8(kernel) => kernel.launch(stream, launch, arguments),
        }?)
    }
}

fn validate(spec: AffineEmbeddingSpec) -> Result<()> {
    let values_per_word = match spec.bits {
        4 => 8,
        8 => 4,
        _ => return Err(Error::InvalidQuantizedGemv("unsupported embedding precision")),
    };
    if spec.vocab == 0
        || spec.hidden == 0
        || spec.group_size == 0
        || !spec.hidden.is_multiple_of(spec.group_size)
        || !spec.hidden.is_multiple_of(values_per_word)
        || !spec.output_scale.is_finite()
        || spec.output_scale <= 0.0
    {
        return Err(Error::InvalidQuantizedGemv("invalid affine embedding geometry"));
    }
    Ok(())
}
