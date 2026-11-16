use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_files,
};

use super::{
    super::geometry::{narrow, product, require},
    GgmlKernelType,
};
use crate::{Error, Result};

macro_rules! ggml_embedding_exports {
    ($($name:ident = $symbol:literal),* $(,)?) => {$(
        cuda_export!($name = $symbol(
            weight: &DeviceBuffer<u8>, selected: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>,
            selected_start: u32, tokens: u32, vocab: u32, hidden: u32, output_scale: f32,
        ));
    )*};
}

ggml_embedding_exports!(
    EmbeddingQ40 = "libmir_cuda_ggml_embedding_bf16_q4_0",
    EmbeddingQ41 = "libmir_cuda_ggml_embedding_bf16_q4_1",
    EmbeddingQ50 = "libmir_cuda_ggml_embedding_bf16_q5_0",
    EmbeddingQ51 = "libmir_cuda_ggml_embedding_bf16_q5_1",
    EmbeddingQ80 = "libmir_cuda_ggml_embedding_bf16_q8_0",
    EmbeddingQ3K = "libmir_cuda_ggml_embedding_bf16_q3_k",
    EmbeddingQ2K = "libmir_cuda_ggml_embedding_bf16_q2_k",
    EmbeddingQ4K = "libmir_cuda_ggml_embedding_bf16_q4_k",
    EmbeddingQ5K = "libmir_cuda_ggml_embedding_bf16_q5_k",
    EmbeddingQ6K = "libmir_cuda_ggml_embedding_bf16_q6_k",
    EmbeddingIQ4NL = "libmir_cuda_ggml_embedding_bf16_iq4_nl",
    EmbeddingIQ4XS = "libmir_cuda_ggml_embedding_bf16_iq4_xs",
    EmbeddingIQ2XXS = "libmir_cuda_ggml_embedding_bf16_iq2_xxs",
    EmbeddingIQ2XS = "libmir_cuda_ggml_embedding_bf16_iq2_xs",
    EmbeddingIQ2S = "libmir_cuda_ggml_embedding_bf16_iq2_s",
    EmbeddingIQ3XXS = "libmir_cuda_ggml_embedding_bf16_iq3_xxs",
    EmbeddingIQ3S = "libmir_cuda_ggml_embedding_bf16_iq3_s",
    EmbeddingIQ1S = "libmir_cuda_ggml_embedding_bf16_iq1_s",
    EmbeddingIQ1M = "libmir_cuda_ggml_embedding_bf16_iq1_m",
    EmbeddingMXFP4 = "libmir_cuda_ggml_embedding_bf16_mxfp4",
);

#[derive(Clone, Debug)]
enum Kernel {
    Q4_0(TypedKernel<EmbeddingQ40>),
    Q4_1(TypedKernel<EmbeddingQ41>),
    Q5_0(TypedKernel<EmbeddingQ50>),
    Q5_1(TypedKernel<EmbeddingQ51>),
    Q8_0(TypedKernel<EmbeddingQ80>),
    Q3K(TypedKernel<EmbeddingQ3K>),
    Q2K(TypedKernel<EmbeddingQ2K>),
    Q4K(TypedKernel<EmbeddingQ4K>),
    Q5K(TypedKernel<EmbeddingQ5K>),
    Q6K(TypedKernel<EmbeddingQ6K>),
    IQ4NL(TypedKernel<EmbeddingIQ4NL>),
    IQ4XS(TypedKernel<EmbeddingIQ4XS>),
    IQ2XXS(TypedKernel<EmbeddingIQ2XXS>),
    IQ2XS(TypedKernel<EmbeddingIQ2XS>),
    IQ2S(TypedKernel<EmbeddingIQ2S>),
    IQ3XXS(TypedKernel<EmbeddingIQ3XXS>),
    IQ3S(TypedKernel<EmbeddingIQ3S>),
    IQ1S(TypedKernel<EmbeddingIQ1S>),
    IQ1M(TypedKernel<EmbeddingIQ1M>),
    MXFP4(TypedKernel<EmbeddingMXFP4>),
}

/// Selected vocabulary rows of a GGML embedding table, dequantized to BF16.
#[derive(Clone, Debug)]
pub struct GgmlEmbedding {
    kernel: Kernel,
    kind: GgmlKernelType,
    vocab: usize,
    hidden: usize,
    output_scale: f32,
}

pub struct GgmlEmbeddingOperands<'a> {
    pub weight: &'a DeviceBuffer<u8>,
    pub selected: &'a DeviceBuffer<u32>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl GgmlEmbedding {
    pub fn compile(
        compiler: &Compiler,
        kind: GgmlKernelType,
        vocab: usize,
        hidden: usize,
        output_scale: f32,
    ) -> Result<Self> {
        let block = usize::try_from(kind.ggml_type().block_elements())?;
        if vocab == 0 || hidden == 0 || !hidden.is_multiple_of(block) || !output_scale.is_finite() {
            return Err(Error::InvalidDecoderKernel("invalid GGML embedding geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_files!(
                "ggml_matrix.cu";
                "../../../kernels/ggml/blocks.cuh",
                "../../../kernels/ggml/legacy.cuh",
                "../../../kernels/ggml/k_quants.cuh",
                "../../../kernels/ggml/iq4.cuh",
                "../../../kernels/ggml/grids/iq2.cuh",
                "../../../kernels/ggml/grids/iq13.cuh",
                "../../../kernels/ggml/iq.cuh",
                "../../../kernels/ggml/matrix.cu",
            ),
            &CompileOptions::default(),
        )?;
        let kernel = match kind {
            GgmlKernelType::Q4_0 => Kernel::Q4_0(module.kernel()?),
            GgmlKernelType::Q4_1 => Kernel::Q4_1(module.kernel()?),
            GgmlKernelType::Q5_0 => Kernel::Q5_0(module.kernel()?),
            GgmlKernelType::Q5_1 => Kernel::Q5_1(module.kernel()?),
            GgmlKernelType::Q8_0 => Kernel::Q8_0(module.kernel()?),
            GgmlKernelType::Q3K => Kernel::Q3K(module.kernel()?),
            GgmlKernelType::Q2K => Kernel::Q2K(module.kernel()?),
            GgmlKernelType::Q4K => Kernel::Q4K(module.kernel()?),
            GgmlKernelType::Q5K => Kernel::Q5K(module.kernel()?),
            GgmlKernelType::Q6K => Kernel::Q6K(module.kernel()?),
            GgmlKernelType::IQ4NL => Kernel::IQ4NL(module.kernel()?),
            GgmlKernelType::IQ4XS => Kernel::IQ4XS(module.kernel()?),
            GgmlKernelType::IQ2XXS => Kernel::IQ2XXS(module.kernel()?),
            GgmlKernelType::IQ2XS => Kernel::IQ2XS(module.kernel()?),
            GgmlKernelType::IQ2S => Kernel::IQ2S(module.kernel()?),
            GgmlKernelType::IQ3XXS => Kernel::IQ3XXS(module.kernel()?),
            GgmlKernelType::IQ3S => Kernel::IQ3S(module.kernel()?),
            GgmlKernelType::IQ1S => Kernel::IQ1S(module.kernel()?),
            GgmlKernelType::IQ1M => Kernel::IQ1M(module.kernel()?),
            GgmlKernelType::MXFP4 => Kernel::MXFP4(module.kernel()?),
        };
        Ok(Self {
            kernel,
            kind,
            vocab,
            hidden,
            output_scale,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        operands: &mut GgmlEmbeddingOperands<'_>,
        selected_start: usize,
        tokens: usize,
    ) -> Result<()> {
        let selected_end = selected_start
            .checked_add(tokens)
            .ok_or(Error::InvalidDecoderKernel("GGML selected range overflow"))?;
        require("GGML selected tokens", selected_end, operands.selected.len())?;
        require(
            "GGML embedding weight",
            self.kind.bytes(self.vocab, self.hidden)?,
            operands.weight.len(),
        )?;
        require("GGML embedding output", product(tokens, self.hidden)?, operands.output.len())?;
        let config = LaunchConfig {
            grid: (narrow((self.hidden / 32).div_ceil(128))?, narrow(tokens)?, 1),
            block: (128, 1, 1),
            shared_memory_bytes: 0,
        };
        let arguments = (
            operands.weight,
            operands.selected,
            &mut *operands.output,
            narrow(selected_start)?,
            narrow(tokens)?,
            narrow(self.vocab)?,
            narrow(self.hidden)?,
            self.output_scale,
        );
        match &self.kernel {
            Kernel::Q4_0(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q4_1(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q5_0(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q5_1(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q8_0(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q3K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q2K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q4K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q5K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q6K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ4NL(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ4XS(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ2XXS(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ2XS(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ2S(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ3XXS(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ3S(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ1S(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ1M(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::MXFP4(kernel) => kernel.launch(stream, config, arguments)?,
        }
        Ok(())
    }
}
