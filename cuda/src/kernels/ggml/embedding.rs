use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Module, Stream, TypedKernel, bf16,
    cuda_export, cuda_kernel_files,
};
use models::gguf::GgmlType;

use super::{
    super::geometry::{narrow, product, require},
    GgmlKernelType,
};
use crate::{Error, Result};

type Arguments<'a> = (
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u32>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    f32,
);

macro_rules! embedding_kernels {
    ($($variant:ident = $symbol:literal, $kind:ident;)*) => {
        $(cuda_export!($variant = $symbol(
            weight: &DeviceBuffer<u8>, selected: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>,
            selected_start: u32, tokens: u32, vocab: u32, hidden: u32, output_scale: f32,
        ));)*

        #[derive(Clone, Debug)]
        enum Kernel {
            $($variant(TypedKernel<$variant>),)*
        }

        impl Kernel {
            fn load(module: &Module, kind: GgmlKernelType) -> Result<Self> {
                match kind.ggml_type() {
                    $(GgmlType::$kind => Ok(Self::$variant(module.kernel()?)),)*
                    _ => Err(Error::InvalidDecoderKernel("no GGML kernel for this type")),
                }
            }

            fn launch(&self, stream: &Stream, config: LaunchConfig, arguments: Arguments<'_>) -> Result<()> {
                match self {
                    $(Self::$variant(kernel) => kernel.launch(stream, config, arguments)?,)*
                }
                Ok(())
            }
        }
    };
}

embedding_kernels! {
    Q40 = "libmir_cuda_ggml_embedding_bf16_q4_0", Q4_0;
    Q41 = "libmir_cuda_ggml_embedding_bf16_q4_1", Q4_1;
    Q50 = "libmir_cuda_ggml_embedding_bf16_q5_0", Q5_0;
    Q51 = "libmir_cuda_ggml_embedding_bf16_q5_1", Q5_1;
    Q80 = "libmir_cuda_ggml_embedding_bf16_q8_0", Q8_0;
    Q2K = "libmir_cuda_ggml_embedding_bf16_q2_k", Q2_K;
    Q3K = "libmir_cuda_ggml_embedding_bf16_q3_k", Q3_K;
    Q4K = "libmir_cuda_ggml_embedding_bf16_q4_k", Q4_K;
    Q5K = "libmir_cuda_ggml_embedding_bf16_q5_k", Q5_K;
    Q6K = "libmir_cuda_ggml_embedding_bf16_q6_k", Q6_K;
    IQ4NL = "libmir_cuda_ggml_embedding_bf16_iq4_nl", IQ4_NL;
    IQ4XS = "libmir_cuda_ggml_embedding_bf16_iq4_xs", IQ4_XS;
    IQ2XXS = "libmir_cuda_ggml_embedding_bf16_iq2_xxs", IQ2_XXS;
    IQ2XS = "libmir_cuda_ggml_embedding_bf16_iq2_xs", IQ2_XS;
    IQ2S = "libmir_cuda_ggml_embedding_bf16_iq2_s", IQ2_S;
    IQ3XXS = "libmir_cuda_ggml_embedding_bf16_iq3_xxs", IQ3_XXS;
    IQ3S = "libmir_cuda_ggml_embedding_bf16_iq3_s", IQ3_S;
    IQ1S = "libmir_cuda_ggml_embedding_bf16_iq1_s", IQ1_S;
    IQ1M = "libmir_cuda_ggml_embedding_bf16_iq1_m", IQ1_M;
    MXFP4 = "libmir_cuda_ggml_embedding_bf16_mxfp4", MXFP4;
    NVFP4 = "libmir_cuda_ggml_embedding_bf16_nvfp4", NVFP4;
    Q10 = "libmir_cuda_ggml_embedding_bf16_q1_0", Q1_0;
    Q20 = "libmir_cuda_ggml_embedding_bf16_q2_0", Q2_0;
    TQ10 = "libmir_cuda_ggml_embedding_bf16_tq1_0", TQ1_0;
    TQ20 = "libmir_cuda_ggml_embedding_bf16_tq2_0", TQ2_0;
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
                "../../../kernels/ggml/low_bit.cuh",
                "../../../kernels/ggml/grids/iq2.cuh",
                "../../../kernels/ggml/grids/iq13.cuh",
                "../../../kernels/ggml/iq.cuh",
                "../../../kernels/ggml/matrix.cu",
            ),
            &CompileOptions::default(),
        )?;
        let kernel = Kernel::load(&module, kind)?;
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
        self.kernel.launch(stream, config, arguments)
    }
}
