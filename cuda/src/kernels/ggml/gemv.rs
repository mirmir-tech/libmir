//! The GEMV exports of `kernels/ggml/vector.cu`: one per GGML type and pass
//! width, all with one signature.

use mircuda::{DeviceBuffer, LaunchConfig, Module, Stream, TypedKernel, bf16, cuda_export};

use super::GgmlKernelType;
use crate::{Error, Result};

/// Pass widths with an export; a batch takes the narrowest that fits it.
pub(super) const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];

pub(super) struct GemvArguments<'a> {
    pub(super) quantized: &'a DeviceBuffer<u32>,
    pub(super) scales: &'a DeviceBuffer<f32>,
    pub(super) weight: &'a DeviceBuffer<u8>,
    pub(super) output: &'a mut DeviceBuffer<bf16>,
    pub(super) tokens: u32,
    pub(super) rows: u32,
    pub(super) columns: u32,
}

macro_rules! gemv_kernels {
    ($($variant:ident = $symbol:literal, $kind:ident, $width:literal;)*) => {
        $(cuda_export!(pub(super) $variant = $symbol(
            quantized: &DeviceBuffer<u32>, scales: &DeviceBuffer<f32>, weight: &DeviceBuffer<u8>,
            output: &mut DeviceBuffer<bf16>, tokens: u32, rows: u32, columns: u32,
        ));)*

        #[derive(Clone, Debug)]
        pub(super) enum GemvKernel {
            $($variant(TypedKernel<$variant>),)*
        }

        impl GemvKernel {
            pub(super) fn load(module: &Module, kind: GgmlKernelType, width: usize) -> Result<Self> {
                match (kind, width) {
                    $((GgmlKernelType::$kind, $width) => Ok(Self::$variant(module.kernel()?)),)*
                    _ => Err(Error::InvalidDecoderKernel("no GGML GEMV for this pass width")),
                }
            }

            pub(super) fn launch(
                &self,
                stream: &Stream,
                config: LaunchConfig,
                arguments: GemvArguments<'_>,
            ) -> Result<()> {
                let GemvArguments { quantized, scales, weight, output, tokens, rows, columns } =
                    arguments;
                let arguments = (quantized, scales, weight, output, tokens, rows, columns);
                match self {
                    $(Self::$variant(kernel) => kernel.launch(stream, config, arguments)?,)*
                }
                Ok(())
            }
        }
    };
}

gemv_kernels! {
    Q40T1 = "libmir_cuda_ggml_gemv_bf16_q4_0_t1", Q4_0, 1;
    Q40T2 = "libmir_cuda_ggml_gemv_bf16_q4_0_t2", Q4_0, 2;
    Q40T4 = "libmir_cuda_ggml_gemv_bf16_q4_0_t4", Q4_0, 4;
    Q40T8 = "libmir_cuda_ggml_gemv_bf16_q4_0_t8", Q4_0, 8;
    Q40T16 = "libmir_cuda_ggml_gemv_bf16_q4_0_t16", Q4_0, 16;
    Q41T1 = "libmir_cuda_ggml_gemv_bf16_q4_1_t1", Q4_1, 1;
    Q41T2 = "libmir_cuda_ggml_gemv_bf16_q4_1_t2", Q4_1, 2;
    Q41T4 = "libmir_cuda_ggml_gemv_bf16_q4_1_t4", Q4_1, 4;
    Q41T8 = "libmir_cuda_ggml_gemv_bf16_q4_1_t8", Q4_1, 8;
    Q41T16 = "libmir_cuda_ggml_gemv_bf16_q4_1_t16", Q4_1, 16;
    Q50T1 = "libmir_cuda_ggml_gemv_bf16_q5_0_t1", Q5_0, 1;
    Q50T2 = "libmir_cuda_ggml_gemv_bf16_q5_0_t2", Q5_0, 2;
    Q50T4 = "libmir_cuda_ggml_gemv_bf16_q5_0_t4", Q5_0, 4;
    Q50T8 = "libmir_cuda_ggml_gemv_bf16_q5_0_t8", Q5_0, 8;
    Q50T16 = "libmir_cuda_ggml_gemv_bf16_q5_0_t16", Q5_0, 16;
    Q51T1 = "libmir_cuda_ggml_gemv_bf16_q5_1_t1", Q5_1, 1;
    Q51T2 = "libmir_cuda_ggml_gemv_bf16_q5_1_t2", Q5_1, 2;
    Q51T4 = "libmir_cuda_ggml_gemv_bf16_q5_1_t4", Q5_1, 4;
    Q51T8 = "libmir_cuda_ggml_gemv_bf16_q5_1_t8", Q5_1, 8;
    Q51T16 = "libmir_cuda_ggml_gemv_bf16_q5_1_t16", Q5_1, 16;
    Q80T1 = "libmir_cuda_ggml_gemv_bf16_q8_0_t1", Q8_0, 1;
    Q80T2 = "libmir_cuda_ggml_gemv_bf16_q8_0_t2", Q8_0, 2;
    Q80T4 = "libmir_cuda_ggml_gemv_bf16_q8_0_t4", Q8_0, 4;
    Q80T8 = "libmir_cuda_ggml_gemv_bf16_q8_0_t8", Q8_0, 8;
    Q80T16 = "libmir_cuda_ggml_gemv_bf16_q8_0_t16", Q8_0, 16;
    Q3KT1 = "libmir_cuda_ggml_gemv_bf16_q3_k_t1", Q3K, 1;
    Q3KT2 = "libmir_cuda_ggml_gemv_bf16_q3_k_t2", Q3K, 2;
    Q3KT4 = "libmir_cuda_ggml_gemv_bf16_q3_k_t4", Q3K, 4;
    Q3KT8 = "libmir_cuda_ggml_gemv_bf16_q3_k_t8", Q3K, 8;
    Q3KT16 = "libmir_cuda_ggml_gemv_bf16_q3_k_t16", Q3K, 16;
    Q2KT1 = "libmir_cuda_ggml_gemv_bf16_q2_k_t1", Q2K, 1;
    Q2KT2 = "libmir_cuda_ggml_gemv_bf16_q2_k_t2", Q2K, 2;
    Q2KT4 = "libmir_cuda_ggml_gemv_bf16_q2_k_t4", Q2K, 4;
    Q2KT8 = "libmir_cuda_ggml_gemv_bf16_q2_k_t8", Q2K, 8;
    Q2KT16 = "libmir_cuda_ggml_gemv_bf16_q2_k_t16", Q2K, 16;
    Q4KT1 = "libmir_cuda_ggml_gemv_bf16_q4_k_t1", Q4K, 1;
    Q4KT2 = "libmir_cuda_ggml_gemv_bf16_q4_k_t2", Q4K, 2;
    Q4KT4 = "libmir_cuda_ggml_gemv_bf16_q4_k_t4", Q4K, 4;
    Q4KT8 = "libmir_cuda_ggml_gemv_bf16_q4_k_t8", Q4K, 8;
    Q4KT16 = "libmir_cuda_ggml_gemv_bf16_q4_k_t16", Q4K, 16;
    Q5KT1 = "libmir_cuda_ggml_gemv_bf16_q5_k_t1", Q5K, 1;
    Q5KT2 = "libmir_cuda_ggml_gemv_bf16_q5_k_t2", Q5K, 2;
    Q5KT4 = "libmir_cuda_ggml_gemv_bf16_q5_k_t4", Q5K, 4;
    Q5KT8 = "libmir_cuda_ggml_gemv_bf16_q5_k_t8", Q5K, 8;
    Q5KT16 = "libmir_cuda_ggml_gemv_bf16_q5_k_t16", Q5K, 16;
    Q6KT1 = "libmir_cuda_ggml_gemv_bf16_q6_k_t1", Q6K, 1;
    Q6KT2 = "libmir_cuda_ggml_gemv_bf16_q6_k_t2", Q6K, 2;
    Q6KT4 = "libmir_cuda_ggml_gemv_bf16_q6_k_t4", Q6K, 4;
    Q6KT8 = "libmir_cuda_ggml_gemv_bf16_q6_k_t8", Q6K, 8;
    Q6KT16 = "libmir_cuda_ggml_gemv_bf16_q6_k_t16", Q6K, 16;
    IQ4NLT1 = "libmir_cuda_ggml_gemv_bf16_iq4_nl_t1", IQ4NL, 1;
    IQ4NLT2 = "libmir_cuda_ggml_gemv_bf16_iq4_nl_t2", IQ4NL, 2;
    IQ4NLT4 = "libmir_cuda_ggml_gemv_bf16_iq4_nl_t4", IQ4NL, 4;
    IQ4NLT8 = "libmir_cuda_ggml_gemv_bf16_iq4_nl_t8", IQ4NL, 8;
    IQ4NLT16 = "libmir_cuda_ggml_gemv_bf16_iq4_nl_t16", IQ4NL, 16;
    IQ4XST1 = "libmir_cuda_ggml_gemv_bf16_iq4_xs_t1", IQ4XS, 1;
    IQ4XST2 = "libmir_cuda_ggml_gemv_bf16_iq4_xs_t2", IQ4XS, 2;
    IQ4XST4 = "libmir_cuda_ggml_gemv_bf16_iq4_xs_t4", IQ4XS, 4;
    IQ4XST8 = "libmir_cuda_ggml_gemv_bf16_iq4_xs_t8", IQ4XS, 8;
    IQ4XST16 = "libmir_cuda_ggml_gemv_bf16_iq4_xs_t16", IQ4XS, 16;
    IQ2XXST1 = "libmir_cuda_ggml_gemv_bf16_iq2_xxs_t1", IQ2XXS, 1;
    IQ2XXST2 = "libmir_cuda_ggml_gemv_bf16_iq2_xxs_t2", IQ2XXS, 2;
    IQ2XXST4 = "libmir_cuda_ggml_gemv_bf16_iq2_xxs_t4", IQ2XXS, 4;
    IQ2XXST8 = "libmir_cuda_ggml_gemv_bf16_iq2_xxs_t8", IQ2XXS, 8;
    IQ2XXST16 = "libmir_cuda_ggml_gemv_bf16_iq2_xxs_t16", IQ2XXS, 16;
    IQ2XST1 = "libmir_cuda_ggml_gemv_bf16_iq2_xs_t1", IQ2XS, 1;
    IQ2XST2 = "libmir_cuda_ggml_gemv_bf16_iq2_xs_t2", IQ2XS, 2;
    IQ2XST4 = "libmir_cuda_ggml_gemv_bf16_iq2_xs_t4", IQ2XS, 4;
    IQ2XST8 = "libmir_cuda_ggml_gemv_bf16_iq2_xs_t8", IQ2XS, 8;
    IQ2XST16 = "libmir_cuda_ggml_gemv_bf16_iq2_xs_t16", IQ2XS, 16;
    IQ2ST1 = "libmir_cuda_ggml_gemv_bf16_iq2_s_t1", IQ2S, 1;
    IQ2ST2 = "libmir_cuda_ggml_gemv_bf16_iq2_s_t2", IQ2S, 2;
    IQ2ST4 = "libmir_cuda_ggml_gemv_bf16_iq2_s_t4", IQ2S, 4;
    IQ2ST8 = "libmir_cuda_ggml_gemv_bf16_iq2_s_t8", IQ2S, 8;
    IQ2ST16 = "libmir_cuda_ggml_gemv_bf16_iq2_s_t16", IQ2S, 16;
    IQ3XXST1 = "libmir_cuda_ggml_gemv_bf16_iq3_xxs_t1", IQ3XXS, 1;
    IQ3XXST2 = "libmir_cuda_ggml_gemv_bf16_iq3_xxs_t2", IQ3XXS, 2;
    IQ3XXST4 = "libmir_cuda_ggml_gemv_bf16_iq3_xxs_t4", IQ3XXS, 4;
    IQ3XXST8 = "libmir_cuda_ggml_gemv_bf16_iq3_xxs_t8", IQ3XXS, 8;
    IQ3XXST16 = "libmir_cuda_ggml_gemv_bf16_iq3_xxs_t16", IQ3XXS, 16;
    IQ3ST1 = "libmir_cuda_ggml_gemv_bf16_iq3_s_t1", IQ3S, 1;
    IQ3ST2 = "libmir_cuda_ggml_gemv_bf16_iq3_s_t2", IQ3S, 2;
    IQ3ST4 = "libmir_cuda_ggml_gemv_bf16_iq3_s_t4", IQ3S, 4;
    IQ3ST8 = "libmir_cuda_ggml_gemv_bf16_iq3_s_t8", IQ3S, 8;
    IQ3ST16 = "libmir_cuda_ggml_gemv_bf16_iq3_s_t16", IQ3S, 16;
    IQ1ST1 = "libmir_cuda_ggml_gemv_bf16_iq1_s_t1", IQ1S, 1;
    IQ1ST2 = "libmir_cuda_ggml_gemv_bf16_iq1_s_t2", IQ1S, 2;
    IQ1ST4 = "libmir_cuda_ggml_gemv_bf16_iq1_s_t4", IQ1S, 4;
    IQ1ST8 = "libmir_cuda_ggml_gemv_bf16_iq1_s_t8", IQ1S, 8;
    IQ1ST16 = "libmir_cuda_ggml_gemv_bf16_iq1_s_t16", IQ1S, 16;
    IQ1MT1 = "libmir_cuda_ggml_gemv_bf16_iq1_m_t1", IQ1M, 1;
    IQ1MT2 = "libmir_cuda_ggml_gemv_bf16_iq1_m_t2", IQ1M, 2;
    IQ1MT4 = "libmir_cuda_ggml_gemv_bf16_iq1_m_t4", IQ1M, 4;
    IQ1MT8 = "libmir_cuda_ggml_gemv_bf16_iq1_m_t8", IQ1M, 8;
    IQ1MT16 = "libmir_cuda_ggml_gemv_bf16_iq1_m_t16", IQ1M, 16;
    MXFP4T1 = "libmir_cuda_ggml_gemv_bf16_mxfp4_t1", MXFP4, 1;
    MXFP4T2 = "libmir_cuda_ggml_gemv_bf16_mxfp4_t2", MXFP4, 2;
    MXFP4T4 = "libmir_cuda_ggml_gemv_bf16_mxfp4_t4", MXFP4, 4;
    MXFP4T8 = "libmir_cuda_ggml_gemv_bf16_mxfp4_t8", MXFP4, 8;
    MXFP4T16 = "libmir_cuda_ggml_gemv_bf16_mxfp4_t16", MXFP4, 16;
}
